import torch
import torchvision

class WRNPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(WRNPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.wide_resnet50_2(pretrained=True).conv1.eval())
        blocks.append(torchvision.models.wide_resnet50_2(pretrained=True).bn1.eval())
        blocks.append(torchvision.models.wide_resnet50_2(pretrained=True).relu.eval())
        blocks.append(torchvision.models.wide_resnet50_2(pretrained=True).maxpool.eval())
        blocks.append(torchvision.models.wide_resnet50_2(pretrained=True).layer1.eval())
        blocks.append(torchvision.models.wide_resnet50_2(pretrained=True).layer2.eval())
        blocks.append(torchvision.models.wide_resnet50_2(pretrained=True).layer3.eval())
        blocks.append(torchvision.models.wide_resnet50_2(pretrained=True).layer4.eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[4, 5, 6, 7], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                b,c,h,w = x.shape
                loss += torch.nn.functional.l2_loss(x, y) / (c*h*w)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss