# Adaptive Adversarial AutoAugment (AAAA)
### CMU 11-785 Final Project
Eugene Yang, Jhao-Ting Chen, Pratik Mandlecha, Fangcheng Zou  
Mentor: Hao Chen, Samruddhi Pai  


## Installation
1. Create Anaconda environment
    ```bash
    conda create -n AAAA python=3.7
    conda activate AAAA
    ```
2. Install required packages
    ```bash
    pip install -r requirements.txt
    ```

## Run AAAA
```bash
python train.py --c config/aaaa.yaml
```

### Run AAAA with Discriminator loss
```bash
python train.py --c config/aaaa_discriminator.yaml
```

### Run AAAA with Perceptual loss
```bash
python train.py --c config/aaaa_perceptual.yaml
```

### Run AAAA with Label Preserving loss
```bash
python train.py --c config/aaaa_label_preserving.yaml
```