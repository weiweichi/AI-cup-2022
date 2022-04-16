import torch
import torch.nn as nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 11),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 7), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 7), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 7), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 5), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(512, 512, 5), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*3*3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 14),
            # nn.LogSoftmax(dim=1) # for nn.NLLLoss
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def get_models(model_name, pretrained):
    model = None
    if model_name == "CNN":
        return CNN()
    elif model_name == "Res18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "Res34":
        model = models.resnet34(pretrained=pretrained)

    if model != None:
        fc_inputs = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 14),
            # nn.LogSoftmax(dim=1) # for nn.NLLLoss
        )
        return model
    
    if model_name == "densenet169": # 1687MiB
        model = models.densenet169(pretrained=pretrained)
    elif model_name == "densenet121": # 1519MiB
        model = models.densenet121(pretrained=pretrained)
    elif model_name == "densenet201": 
        model = models.densenet201(pretrained=pretrained)
        
    fc_inputs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(fc_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 14), 
        # nn.LogSoftmax(dim=1) # for nn.NLLLoss
    )
    return model