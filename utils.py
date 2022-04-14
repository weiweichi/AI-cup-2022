import os, glob
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

test_tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, shear=0.3),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class myDataset(Dataset):
    def __init__(self, folder, tfm=test_tfm):
        self.files = sorted([p for dir_path in folder for p in glob.glob(os.path.join(dir_path, "*"))])

        self.transform = tfm

        print(f"One sample", self.files[0])
        print(f"number of samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)
  
    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0]) # as training label
        except:
            label = fname.split("/")[-1].split(".")[0] # as files' name

        return im, label

def get_dataloader(data_dir, batch_size, n_workers, _test = False):
    """Generate dataloader"""
    dataset = myDataset(data_dir)

    if _test:
        return DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    
    # Split dataset into training dataset and validation dataset
    trainlen = int(0.8 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

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