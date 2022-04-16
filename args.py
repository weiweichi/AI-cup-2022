args = {
    "dir_name":[
        "banana", "carrot", 'corn', 'dragonfruit', 'garlic', 'guava', 'peanut',\
        'pineapple', 'pumpkin', 'rice', 'soybean', 'sugarcane', 'tomato', 'bareland', 'inundated'
    ],
    "label2name":{
        0: 'banana', 1: 'carrot', 2: 'corn', 3: 'dragonfruit', 4: 'garlic', 5:'guava',\
        6: 'peanut', 7: 'pineapple', 8: 'pumpkin', 9: 'rice', 10: 'soybean',\
        11: 'sugarcane', 12: 'tomato', 13: 'bareland'#, 14: 'inundated'
    },

    # model type and use pretrained
    "model_name": "densenet169", # CNN, Res18, Res34, densenet169, densenet121 and densenet201
    "pretrained": True,

    # hyperparameters
    "batch_size": 32, # 64 for CNN, 32 for others, becuz cache bomb...
    "n_epoch": 25,   
    "n_workers": 2,
    
    "lr_warmup": 500,  # warm-up step
    "lr": 0.07,    # lr * 550**(-0.5) * min(step ** (-0.5), step * lr_warmup ** (-1.5))
    "weight_decay": 1e-5,
    "clip_norm": 10.0,   # clipping gradient norm helps alleviate gradient exploding

    "swa_lr": 5e-3,    # SWA lr
    "swa_start": 10,   # epoch'th to start SWA

    # folder path
    "save_dir": "./checkpoints", 
    "train_dir": "./data",
    "test_dir": "./test",

    # flag 
    "use_wandb": True,  # record parameter for training
    "train": True,
    "predict": False,
    "use_swa": True,    # implement swa_model to compute average weights
}