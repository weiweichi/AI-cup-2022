args = {
    "dir_name":[
        "banana", "carrot", 'corn', 'dragonfruit', 'garlic', 'guava', 'peanut',\
        'pineapple', 'pumpkin', 'rice', 'soybean', 'sugarcane', 'tomato', 'bareland'#, 'inundated'
    ],
    "label2name":{
        0: 'banana', 1: 'carrot', 2: 'corn', 3: 'dragonfruit', 4: 'garlic', 5:'guava',\
        6: 'peanut', 7: 'pineapple', 8: 'pumpkin', 9: 'rice', 10: 'soybean',\
        11: 'sugarcane', 12: 'tomato', 13: 'bareland',# 14: 'bareland'
    },

    # model type
    "model_name": "densenet169", # CNN, Res18, Res34, Res50, wide_res, densenet169, densenet121 and densenet201
    "pretrained": True,  # use your pretrained model in ./checkpoints. if no model find, ignored

    # hyperparameters
    "batch_size": 32, # 32 for wide_res, Res50 and densenet*, 64 for other, becuz cache boom... BTW, my GPU is 1080ti (10GB)
    "n_workers": 4,
    "n_epoch": 40,
    "swa_start": 8,   # number of epoch to start SWA 
    "swa_lr": 5e-3,    # SWA lr
    
    "lr_warmup": 1000,  # warm-up step
    "lr": 0.003,    # lr * min(step ** (-0.5), step * lr_warmup ** (-1.5))
    "weight_decay": 1e-2,
    "clip_norm": 10,   # clipping gradient norm helps alleviate gradient exploding

    # folder path
    "save_dir": "./checkpoints", 
    "train_dir": "./data",  
    "test_dir": "./test",   

    # flag 
    "use_wandb": True,  # record hyperparameters and acc @ wandb.ai  link: https://wandb.ai/weiweichi/Ai%20cup-2022?workspace=user-weiweichi
    "train": True,
    "predict": False,
    "use_swa": True,    # implement swa_model to compute average weights
    "has_valid": False,  # if True, data will be divided to two parts, one is training data another is validation data.

    # predict
    "models_name": ["densenet169", "Res50", "Res34"]   # predict by fusion, see models.py for more detailed
}