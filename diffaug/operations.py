""" Operations

"""

from typing import Optional, Callable, Tuple
from cv2 import magnitude

import torch
from torch import nn
from torch.distributions import RelaxedBernoulli, Bernoulli, Uniform

from .functional import (
    identity,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    hflip,
    vflip,
    rotate,
    invert,
    solarize,
    posterize,
    gray,
    contrast,
    auto_contrast,
    saturate,
    brightness,
    hue,
    sample_pairing,
    equalize,
    sharpness,
)
from .kernels import get_sharpness_kernel

__all__ = [
    "Identity",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "HorizontalFlip",
    "VerticalFlip",
    "Rotate",
    "Invert",
    "Solarize",
    "Posterize",
    "Gray",
    "Contrast",
    "AutoContrast",
    "Saturate",
    "Brightness",
    "Hue",
    "SamplePairing",
    "Equalize",
    "Sharpness",
]


class _Operation(nn.Module):
    """ Base class of operation

    :param operation:
    :param initial_magnitude:
    :param initial_probability:
    :param magnitude_range:
    :param probability_range:
    :param temperature: Temperature for RelaxedBernoulli distribution used during training
    :param flip_magnitude: Should be True for geometric
    :param debug: If True, check if img image is in [0, 1]
    """

    def __init__(
        self,
        operation: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        requires_magnitude: bool = False,
        requires_probability: bool = False,
        initial_magnitude: Optional[float] = None,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = None,
        probability_range: Optional[Tuple[float, float]] = None,
        temperature: float = 0.1,
        flip_magnitude: bool = False,
        magnitude_scale: float = 1,
        debug: bool = False,
    ):

        super(_Operation, self).__init__()
        self.operation = operation
        assert requires_magnitude ^ requires_probability
        self.requires_magnitude = requires_magnitude
        self.requires_probability = requires_probability

        self.magnitude_range = None
        if not self.requires_magnitude:
            self._magnitude = None
        elif magnitude_range is None:
            self.register_buffer("_magnitude", torch.empty(1).fill_(initial_magnitude))
        else:
            self._magnitude = initial_magnitude
            assert 0 <= magnitude_range[0] < magnitude_range[1] <= 1
            self.magnitude_range = magnitude_range

        self.probability_range = probability_range
        if self.probability_range is None:
            self.register_buffer(
                "_probability", torch.empty(1).fill_(initial_probability)
            )
        else:
            assert 0 <= initial_probability <= 1
            assert 0 <= self.probability_range[0] < self.probability_range[1] <= 1
            self._probability = nn.Parameter(torch.empty(1).fill_(initial_probability))

        assert 0 < temperature
        self.register_buffer("temperature", torch.empty(1).fill_(temperature))

        self.flip_magnitude = flip_magnitude

        assert 0 < magnitude_scale
        self.magnitude_scale = magnitude_scale
        self.debug = debug

    def forward(self, input: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
        """

        :param input: torch.Tensor in [0, 1]
        :return: torch.Tensor in [0, 1]
        """
        if input.dim() == 3:
            input = input.unsqueeze(0)
            mag = mag.unsqueeze(0)
        assert input.dim() == 4
        assert 0 <= input.min() <= input.max() <= 1

        # magnitue
        if self.requires_magnitude:
            if self._magnitude is None:
                mag = self._magnitude
            else:
                if self.magnitude_range is not None:
                    mag = mag.clamp(*self.magnitude_range)

                if self.flip_magnitude and torch.randint(2, (1,)):
                    mag *= -1

                mag = mag * self.magnitude_scale

                return self.operation(input, mag).clamp(0, 1)
        elif self.requires_probability:
            mask = self.get_mask(mag, input.size(0))
            if self.training:
                return (mask * self.operation(input, mag) + (1 - mask) * input).clamp(
                    0, 1
                )
            else:
                mask = mask.squeeze()
                if mask.dim() == 0:
                    mask = mask.unsqueeze(0)
                output = input
                num_valid = mask.sum().long()
                if torch.is_tensor(mag):
                    if mag.size(0) == 1:
                        mag = mag.repeat(num_valid)
                    else:
                        mag = mag[mask == 1]
                if num_valid > 0:
                    output[mask == 1, ...] = self.operation(output[mask == 1, ...], mag)
                return output.clamp(0, 1)

    def get_mask(self, prob: torch.Tensor, batch_size=None) -> torch.Tensor:
        size = (batch_size, 1, 1, 1)
        if self.probability_range is not None:
            prob = prob.clamp(*self.probability_range)
        if self.training:
            return RelaxedBernoulli(self.temperature, prob).rsample().reshape(size)
        else:
            return Bernoulli(prob).sample()


# Geometric Operations

class Identity(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = False,
        requires_probability: bool = True,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(Identity, self).__init__(
            identity,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            debug=debug,
        )

class ShearX(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.05,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        magnitude_scale: float = 1.0,
        debug: bool = False,
    ):
        super(ShearX, self).__init__(
            shear_x,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            flip_magnitude=True,
            magnitude_scale=magnitude_scale,
            debug=debug,
        )


class ShearY(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.05,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        magnitude_scale: float = 1.0,
        debug: bool = False,
    ):
        super(ShearY, self).__init__(
            shear_y,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            flip_magnitude=True,
            magnitude_scale=magnitude_scale,
            debug=debug,
        )


class TranslateX(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        magnitude_scale: float = 0.45,
        debug: bool = False,
    ):
        super(TranslateX, self).__init__(
            translate_x,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            flip_magnitude=True,
            magnitude_scale=magnitude_scale,
            debug=debug,
        )


class TranslateY(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        magnitude_scale: float = 0.45,
        debug: bool = False,
    ):
        super(TranslateY, self).__init__(
            translate_y,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            flip_magnitude=True,
            magnitude_scale=magnitude_scale,
            debug=debug,
        )


class HorizontalFlip(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = False,
        requires_probability: bool = True,
        initial_probability: float = 0.5,
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(HorizontalFlip, self).__init__(
            hflip,
            requires_magnitude,
            requires_probability,
            None,
            initial_probability,
            None,
            probability_range,
            temperature,
            debug=debug,
        )


class VerticalFlip(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = False,
        requires_probability: bool = True,
        initial_probability: float = 0.5,
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(VerticalFlip, self).__init__(
            vflip,
            requires_magnitude,
            requires_probability,
            None,
            initial_probability,
            None,
            probability_range,
            temperature,
            debug=debug,
        )


class Rotate(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        magnitude_scale: float = 30,
        debug: bool = False,
    ):
        super(Rotate, self).__init__(
            rotate,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            flip_magnitude=True,
            magnitude_scale=magnitude_scale,
            debug=debug,
        )


# Color Enhancing Operations


class Invert(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = False,
        requires_probability: bool = True,
        initial_probability: float = 0.5,
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(Invert, self).__init__(
            invert,
            requires_magnitude,
            requires_probability,
            None,
            initial_probability,
            None,
            probability_range,
            temperature,
            debug=debug,
        )


class Solarize(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(Solarize, self).__init__(
            solarize,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            debug=debug,
        )


class Posterize(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(Posterize, self).__init__(
            posterize,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            debug=debug,
        )


class Gray(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = False,
        requires_probability: bool = True,
        initial_probability: float = 0.5,
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(Gray, self).__init__(
            gray,
            requires_magnitude,
            requires_probability,
            None,
            initial_probability,
            None,
            probability_range,
            temperature,
            debug=debug,
        )


class Contrast(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(Contrast, self).__init__(
            contrast,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            flip_magnitude=True,
            debug=debug,
        )


class AutoContrast(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = False,
        requires_probability: bool = True,
        initial_probability: float = 0.5,
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(AutoContrast, self).__init__(
            auto_contrast,
            requires_magnitude,
            requires_probability,
            None,
            initial_probability,
            None,
            probability_range,
            temperature,
            debug=debug,
        )


class Saturate(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(Saturate, self).__init__(
            saturate,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            flip_magnitude=True,
            debug=debug,
        )


class Brightness(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(Brightness, self).__init__(
            brightness,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            flip_magnitude=True,
            debug=debug,
        )


class Hue(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        magnitude_scale: float = 2,
        debug: bool = False,
    ):
        super(Hue, self).__init__(
            hue,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            magnitude_scale=magnitude_scale,
            debug=debug,
        )


class SamplePairing(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(SamplePairing, self).__init__(
            sample_pairing,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            debug=debug,
        )


class Equalize(_Operation):
    def __init__(
        self,
        requires_magnitude: bool = False,
        requires_probability: bool = True,
        initial_probability: float = 0.5,
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(Equalize, self).__init__(
            equalize,
            requires_magnitude,
            requires_probability,
            None,
            initial_probability,
            None,
            probability_range,
            temperature,
            debug=debug,
        )


class _KernelOperation(_Operation):
    def __init__(
        self,
        operation: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        kernel: torch.Tensor,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        flip_magnitude: bool = False,
        magnitude_scale: float = 1,
        debug: bool = False,
    ):
        super(_KernelOperation, self).__init__(
            None,
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            flip_magnitude=flip_magnitude,
            magnitude_scale=magnitude_scale,
            debug=debug,
        )

        # to use kernel properly, this is an ugly way...
        self.register_buffer("kernel", kernel)
        self._original_operation = operation
        self.operation = self._operation

    def _operation(self, img: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
        return self._original_operation(img, mag, self.kernel)


class Sharpness(_KernelOperation):
    def __init__(
        self,
        requires_magnitude: bool = True,
        requires_probability: bool = False,
        initial_magnitude: float = 0.5,
        initial_probability: float = 0.5,
        magnitude_range: Optional[Tuple[float, float]] = (0, 1),
        probability_range: Optional[Tuple[float, float]] = (0, 1),
        temperature: float = 0.1,
        debug: bool = False,
    ):
        super(Sharpness, self).__init__(
            sharpness,
            get_sharpness_kernel(),
            requires_magnitude,
            requires_probability,
            initial_magnitude,
            initial_probability,
            magnitude_range,
            probability_range,
            temperature,
            flip_magnitude=True,
            debug=debug,
        )

