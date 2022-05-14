import numpy as np
import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

# import my function
import utils   
import models
from args import args

if __name__ == '__main__':
    
    myseed = 10942178  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
    
    if args['train']:
        if args["use_wandb"]:
            import wandb
            wandb.init(project='Ai cup-2022', config=args, name=args['model_name'])
        # create checkpoints folder
        if not os.path.isdir(args["save_dir"]):
            os.mkdir(args["save_dir"])
        if not os.path.isdir(os.path.join(args["save_dir"], args["model_name"])):
            os.mkdir(os.path.join(args["save_dir"], args["model_name"]))

        root = args["train_dir"]
        folders = [os.path.join(root, p) for p in args['dir_name']]
        train_loader, valid_loader = utils.get_dataloader(folders=folders, batch_size=args["batch_size"], n_workers=args["n_workers"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.get_models(model_name=args["model_name"], pretrained=args["pretrained"])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr = args["lr"], weight_decay=args['weight_decay'])
        scheduler = LambdaLR(optimizer, utils.get_exp_lr_with_warmup)

        swa_model = None
        swa_scheduler = None
        if args['use_swa']:
            swa_model = optim.swa_utils.AveragedModel(model) # something like ensemble 
            swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=args["swa_lr"], anneal_epochs=5)
        swa_flag = False # When to start to use swa_model

        best_acc = 0.0
        for epoch in range(1, args["n_epoch"]+1):

            if args['use_swa']:
                swa_flag = epoch >= args['swa_start']

            # ----------- Training ------------
            train_loss, train_acc = utils.train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler, swa_flag, swa_scheduler, swa_model, epoch)

            # ---------- Validation ----------
            if args['has_valid']:
                valid_acc, valid_loss, best_acc = utils.evaluate(model, valid_loader, criterion, device, epoch, best_acc)

                if args['use_wandb']:
                    wandb.log({
                        "epoch/epoch": epoch,
                        "epoch/valid_acc": valid_acc,
                        "epoch/valid_loss": valid_loss,
                        "epoch/train_acc": train_acc,
                        "epoch/train_loss": train_loss,
                    })

        if args['use_swa']:
            print("Testing for SWA model...")
            try:
                optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            except:
                print("can't not update bn...")
                
            if args['has_valid']:
                valid_acc, valid_loss, best_acc = utils.evaluate(swa_model, valid_loader, criterion, device, epoch, best_acc)
                print(f"[ SWA valid ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
                if args['use_wandb']:
                    wandb.log({
                        "swa_acc": valid_acc,
                        "swa_loss": valid_loss,
                    })

            torch.save(swa_model.state_dict(), "{}/{}/swa.ckpt".format(args["save_dir"], args["model_name"]))
            print('Saving SWA model as {}/{}/swa.ckpt'.format(args["save_dir"], args["model_name"]))
            
        print("Finish training!")

    if args['predict']:
        utils.predict_by_fusion()
        print("Finish predicting!")