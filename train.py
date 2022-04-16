import numpy as np
import os
import torch
from torch import nn, optim

from tqdm.auto import tqdm

from torch.optim.lr_scheduler import LambdaLR

# import my function
import utils   
import models
from args import args


def train():
    if not os.path.isdir(args["save_dir"]):
        os.mkdir(args["save_dir"])

    root = args["train_dir"]
    data_dir = [os.path.join(root, p) for p in args['dir_name']]
    train_loader, valid_loader = utils.get_dataloader(data_dir=data_dir, batch_size=args["batch_size"], n_workers=args["n_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.get_models(model_name=args["model_name"], pretrained=args["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = args["lr"])
    scheduler = LambdaLR(optimizer, get_exp_lr_with_warmup)

    if args['use_swa']:
        swa_model = optim.swa_utils.AveragedModel(model) # something like ensemble 
        swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=0.001, anneal_epochs=10)
        swa_flag = False # start to use swa_model

    best_acc = 0.0

    for epoch in range(args["n_epoch"]):

        # ---------- Training ----------
        model.train()
        train_loss = []
        train_accs = []

        # switch to the SWA
        if args['use_swa']:
            swa_flag = epoch >= args['swa_start']

        for imgs, labels in tqdm(train_loader):
            
            logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))

            # update model
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args["clip_norm"])
            optimizer.step()
            if not swa_flag:
                scheduler.step()
            optimizer.zero_grad()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            
            if args["use_wandb"]:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/acc": acc,
                    "train/lr": optimizer.param_groups[0]['lr']
                })
        # update swa_model
        if swa_flag:
            swa_scheduler.step()
            swa_model.update_parameters(model)

            swa_model.eval()

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
    
        print("[ Train: {:02d}/{:02d} ] loss = {:.5f}, acc = {:.5f}".format(
            epoch,
            args["n_epoch"],
            train_loss,
            train_acc
        ))

        # ---------- Validation ----------
        model.eval()

        valid_loss = []
        valid_accs = []

        for imgs, labels in tqdm(valid_loader):
            if swa_flag:
                with torch.no_grad():
                    logits = swa_model(imgs.to(device))
            else:
                with torch.no_grad():
                    logits = model(imgs.to(device))
            
            loss = criterion(logits, labels.to(device))

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)
        
        
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        if args["use_wandb"]:
            wandb.log({
                "valid/acc": valid_acc,
                "epoch": epoch
            })
            
        if valid_acc > best_acc:
            if swa_flag:
                torch.save(swa_model.state_dict(), "{}/{}_swa.ckpt".format(args["save_dir"], args["model_name"]))
            else:
                torch.save(model.state_dict(), "{}/{}_best.ckpt".format(args["save_dir"], args["model_name"])) 
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
    if args['use_swa']:
        optim.swa_utils.update_bn(train_loader, swa_model)
        torch.save(swa_model.state_dict(), "{}/{}_swa.ckpt".format(args["save_dir"], args["model_name"]))

def predict():
    folder = ["./test"]
    test_loader = utils.get_dataloader(data_dir=folder, batch_size=args["batch_size"], n_workers=args["n_workers"], _test = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.get_models(model_name=args["model_name"], pretrained=args["pretrained"]).to(device)

    if args['use_swa']:
        model.load_state_dict(torch.load("./checkpoints/{}_swa.ckpt".format(args['model_name'])))
    else:
        model.load_state_dict(torch.load("./checkpoints/{}_best.ckpt".format(args['model_name'])))

    model.eval()

    print('predicting...')
    pred = []
    with torch.no_grad():
        for imgs, files_name in tqdm(test_loader):
            features = imgs
            features = features.to(device)

            outputs = model(features)

            _, test_preds = torch.max(outputs, 1) # get the index of the class with the highest probability

            for pair in zip(files_name, test_preds.cpu()):
                pred += [pair] # pair = file_name, test_pred

    print("saving prediction...")
    label2name = args['label2name']
    with open('prediction_{}.csv'.format(args['model_name']), 'w') as f:
        # f.write('Id,Class\n')
        for name, type in tqdm(pred):
            f.write('{},{}\n'.format(name, label2name[type]))
    print("finish predicting! save prediction as prediction_{}.csv".format(args['model_name']))

def get_rate(step_num, warmup_step):
    return 550**(-0.5) * min(step_num ** (-0.5), step_num * warmup_step ** (-1.5))

def get_exp_lr_with_warmup(step):
    return 1e-10 if step == 0 else args["lr_factor"] * get_rate(step, args["lr_warmup"])

if __name__ == '__main__':
    if args["use_wandb"]:
        import wandb
        wandb.init(project='Ai cup-2022', config=args)
    
    myseed = 10942178  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
    
    if args['train']:
        train()
        print("Finish training!")

    if args['predict']:
        predict()
        print("Finish predicting!")