import os, glob
from typing import Callable
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
import torchvision
import wandb
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from args import args
import models

test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.Resize((224, 224)),
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
        if self._test:
            label = fname.split("/")[-1] # as name of data
        else:
            label = int(fname.split("/")[-1].split("_")[0]) # as training label

        return im, label
    
    # for ImbalancedDatasetSampler
    def get_labels(self):
        labels = [fname.split("/")[-1] for fname in self.files]
        return labels

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
        sampler=ImbalancedDatasetSampler(trainset),
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

# https://github.com/ufoym/imbalanced-dataset-sampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

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

def predict_by_fusion():
    _models_list = args['models_list']
    if len(_models_list) == 0:
        print("You don't choose any model!\nChoose the model you wanna use to predict!")
        raise OSError
    
    print("loading test data...")
    folders = [os.path.join('test', f'test_{i}') for i in '0123456789abcdef']
    test_loader = get_dataloader(folders=folders, batch_size=16, n_workers=args["n_workers"], _test = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading model with", *_models_list, '...')
    model_list = []
    for name in _models_list:
        model = models.get_models(model_name=name, pretrained=False)
        model_list.append(model.eval())
        print("-----------------------")
    print(f'model num: {len(model_list)}')
    print('predicting ...')
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
    with open('prediction.csv'.format(args['model_name']), 'w') as f:
        f.write('image_filename,label\n')
        for name, type in tqdm(pred):
            f.write('{},{}\n'.format(name, label2name[int(type)]))
    print("Finish predicting! save prediction as prediction.csv")