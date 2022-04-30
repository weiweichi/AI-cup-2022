# Ai cup 2022 Image Classification
[link](https://aidea-web.tw/topic/93c8c26b-0e96-44bc-9a53-1c96353ad340)

<!-- ---
## Before Run  
*   Download data and unzip by yourself.
*   You should save it in correct folder. -->
---
## Run code
*   First, run `preprocess.py`, it will rename training data.
*   Then, run `train.py` by settings in `args.py`.
*   You can set your hyperparameters in `args.py`.
## Settings
|model          |Res34 |CNN   |densenet169|Res50  |
|:-------------:|:----:|:----:|:---------:|:-----:|
|**epoch**      |8+32  |      |8+32       |
|**batch_size** |64    |64    |32         |32     |
|**lr**         |0.0025|0.0025|0.003      |
|**lr_warmup**  |500   |500   |1000       |
|**swa_lr**     |3e-3  |3e-3  |5e-3       |
|**pretrained** |True  |  -   |True       |


Reference: [ResNet](https://arxiv.org/abs/1512.03385), [DenseNet](https://arxiv.org/abs/1608.06993), [SWA](https://arxiv.org/abs/1803.05407)