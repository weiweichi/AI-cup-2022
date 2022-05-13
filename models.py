import torch
import torch.nn as nn
from torchvision import models
from args import args

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 7), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 5), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 5), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(512, 1024, 3), 
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 14),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def get_models(model_name, pretrained):
    # pretrained model path
    model_path ="{}/{}/ckpt_best.ckpt".format(args["save_dir"], model_name)
    model = None
    if model_name == "CNN":
        model = CNN()
        if pretrained:
            try:
                model.load_state_dict(torch.load(model_path))
                print("pretrained model is loaded!")
            except:
                print(f"pretrained model doesn't exist! model path: {model_path}")
            if args['use_swa']:
                try:
                    model.load_state_dict(torch.load("{}/{}/swa.ckpt".format(args["save_dir"], model_name)))
                except:
                    pass
        return model
    elif model_name == "Res18":
        model = models.resnet18(pretrained=True)
    elif model_name == "Res34":
        model = models.resnet34(pretrained=True)
    elif model_name == "Res50":
        model = models.resnet50(pretrained=True)
    elif model_name == "wide_res":
        model = models.wide_resnet50_2(pretrained=True)

    if model != None:
        fc_inputs = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 14),
        )
        if pretrained:
            try:
                model.load_state_dict(torch.load(model_path))
                print("pretrained model is loaded!")
            except:
                print("pretrained model doesn't exist!")
            if args['use_swa']:
                try:
                    model.load_state_dict(torch.load("{}/{}/swa.ckpt".format(args["save_dir"], model_name)))
                except:
                    pass
        return model
    
    if model_name == "densenet169":
        model = models.densenet169(pretrained=True)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
    elif model_name == "densenet201": 
        model = models.densenet201(pretrained=True)
        
    fc_inputs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(fc_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 14), 
    )
    if pretrained:
        try:
            model.load_state_dict(torch.load(model_path))
            print("pretrained model is loaded!")
        except:
            print("pretrained model doesn't exist!")
        if args['use_swa']:
            try:
                model.load_state_dict(torch.load("{}/{}/swa.ckpt".format(args["save_dir"], model_name)))
            except:
                pass
    return model