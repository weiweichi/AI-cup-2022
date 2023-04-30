# Ai cup 2022-spring Image Classification
A competition organized by AIdea with `428` teams participating lasted for 2 months.   
Our job is classfying agricultural products. I tried ResNet, DenseNet and coatnet models. Total training 42 epochs, warm-up for 10 epochs.     
Also, I implemented SWA to avoid overfitting. But, the accuracy of testing data is 99% already.   
I end up ranking `23rd`.   
[Contest link](https://aidea-web.tw/topic/93c8c26b-0e96-44bc-9a53-1c96353ad340)   
[Certificate](https://global.turingcerts.com/co/cert?hash=c26e6569477cb6e7d0e1e07dfaf4b86edd14ed8c77f3ee91c50341436dae4036)

---
## Run code
*   First, run `preprocess.py`, it will rename training data.
*   Then, run `train.py` by settings in `args.py`.
*   You can set your hyperparameters in `args.py`.
## Settings
|     model      | Res34  |  CNN   | densenet169 | Res50  | coatnet |
| :------------: | :----: | :----: | :---------: | :----: | :-----: |
|   **epoch**    | 10+32  |        |    10+32    | 10+32  |  10+32  |
| **batch_size** |   64   |   64   |     32      |   32   |   32    |
|     **lr**     | 0.0025 | 0.0025 |   0.0025    | 0.0025 | 0.0025  |
| **lr_warmup**  |  500   |  500   |     500     |  500   |   500   |
|   **swa_lr**   |  5e-3  |  5e-3  |    5e-3     |  5e-3  |  5e-3   |
| **pretrained** |  True  |   -    |    True     |  True  |    -    |


Reference: [ResNet](https://arxiv.org/abs/1512.03385), [DenseNet](https://arxiv.org/abs/1608.06993), [SWA](https://arxiv.org/abs/1803.05407), [ImbalancedDatasetSampler](https://github.com/ufoym/imbalanced-dataset-sampler)