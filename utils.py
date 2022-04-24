import os, glob
from PIL import Image
from tqdm.auto import tqdm

import wandb
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
croptype2label = {
         'banana':0,  'carrot' : 1, 'corn' : 2,  'dragonfruit':3,  'garlic':4, 'guava':5,\
        'peanut':6,  'pineapple':7, 'pumpkin':8,  'rice':9,  'soybean':10,\
         'sugarcane':11,  'tomato':12,  'bareland':13, 'inundated':14
    }

from args import args
import models

test_tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class myDataset(Dataset):
    def __init__(self, files_path, tfm=test_tfm, _test = False):
        self.files = files_path
        self._test = _test
        self.transform = tfm

        print(f"One sample", self.files[0])

    def __len__(self):
        return len(self.files)
  
    def __getitem__(self, idx):
        fname = self.files[idx]

        im = Image.open(fname)
        im = self.transform(im)

        # croptype = (fname.split("/")[-2]) # as training label
        # label = croptype2label[croptype]

        if self._test:
            label = fname.split("/")[-1] # as name of data
        else:
            label = int(fname.split("/")[-1].split("_")[0]) # as training label


        return im, label

def get_dataloader(folders, batch_size, n_workers, _test = False):
    """Generate dataloader"""
    files_path = sorted([p for dir_path in folders for p in glob.glob(os.path.join(dir_path, "*"))])
    print(f"number of total samples: {len(files_path)}")

    # testset
    if _test:
        dataset = myDataset(files_path, _test=_test)
        return DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    # only create trainset
    if not args['has_valid']:
        dataset = myDataset(files_path, train_tfm)
        return DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True, shuffle=True), None
    
    # Split dataset into training dataset and validation dataset
    trainlen = int(0.8 * len(files_path))
    lengths = [trainlen, len(files_path) - trainlen]
    trainset, validset = random_split(files_path, lengths)
    trainset, validset = myDataset(trainset, train_tfm), myDataset(validset)
    print(f"number of train samples: {len(trainset)}, number of valid samples: {len(validset)}")
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
    )
    return train_loader, valid_loader

def get_rate(step_num):
    return min(step_num ** (-0.5), step_num * args["lr_warmup"] ** (-1.5))

def get_exp_lr_with_warmup(step):
    return get_rate(step) if step else 1e-10

def train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler, swa_flag, swa_scheduler, swa_model, epoch):
    model = model.to(device)
    model.train()
    train_loss = []
    train_accs = []
    for imgs, labels in tqdm(train_loader):
        logits = model(imgs.to(device))

        loss = criterion(logits, labels.to(device))

        # update model
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args["clip_norm"])
        optimizer.step()

        if not swa_flag:
            scheduler.step()

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

        # plot at web: wandb.ai
        if args['use_wandb']:
            wandb.log({
                "train/loss": loss.item(),        
                "train/grad_norm": grad_norm,
                "train/acc": acc,               
                "train/lr": optimizer.param_groups[0]['lr']
            })

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    if swa_flag:
        swa_scheduler.step()
        swa_model.update_parameters(model)

    print("[ Train: {:02d}/{:02d} ] loss = {:.5f}, acc = {:.5f}".format(
            epoch,
            args["n_epoch"],
            train_loss,
            train_acc
        ))
    
    torch.save(model.state_dict(), "{}/{}/ckpt_{}.ckpt".format(args["save_dir"], args["model_name"], epoch)) 
    return train_loss, train_acc

def evaluate(model, data_loader, criterion, device, epoch, best_acc):
    model = model.to(device)
    model.eval()

    valid_loss = []
    valid_accs = []
    for imgs, labels in tqdm(data_loader):
        with torch.no_grad():
            logits = model(imgs.to(device))
        
        loss = criterion(logits, labels.to(device))

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_accs.append(acc)
    
    
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # -------- print and save ---------
    if valid_acc > best_acc:
        torch.save(model.state_dict(), "{}/{}/ckpt_best.ckpt".format(args["save_dir"], args["model_name"])) 
        best_acc = valid_acc
        print("[ Valid: {:02d}/{:02d} ] loss = {:.5f}, acc = {:.5f} -> best".format(
            epoch,
            args["n_epoch"],
            valid_loss,
            valid_acc
        ))
    else:
        print("[ Valid: {:02d}/{:02d} ] loss = {:.5f}, acc = {:.5f}".format(
            epoch,
            args["n_epoch"],
            valid_loss,
            valid_acc
        ))
    return valid_acc, valid_loss, best_acc

def predict_by_fusion(models_name: list):
    if len(models_name) == 0:
        print("You don't choose any model!\nChoose the model you wanna use to predict!")
        raise ValueError
    
    print("loading test data...")
    folder = ["./test"]
    test_loader = get_dataloader(data_dir=folder, batch_size=16, n_workers=args["n_workers"], _test = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading model...")
    model_list = []
    for name in models_name:
        model = models.get_models(model_name=name, pretrained=True)
        model_list.append(model.eval())
    
    print('predicting...')
    pred = []
    with torch.no_grad():
        for imgs, files_name in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = None
            for model in model_list:
                model = model.to(device)
                if outputs == None:
                    outputs = model(imgs)
                else:
                    outputs += model(imgs)

            outputs /= len(model_list)
            _, test_preds = torch.max(outputs, 1) # get the index of the class with the highest probability

            for name, label in zip(files_name, test_preds.cpu()):
                pred += [(name, label)] # pair = file_name, test_pred
                
    print("saving prediction...")
    label2name = args['label2name']
    with open('prediction_{}.csv'.format(args['model_name']), 'w') as f:
        # f.write('Id,Class\n')
        for name, type in tqdm(pred):
            f.write('{},{}\n'.format(name, label2name[int(type)]))
    print("Finish predicting! save prediction as prediction_{}.csv".format(args['model_name']))