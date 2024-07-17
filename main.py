import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
import itertools

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_acc = None
        self.counter = 0

    def __call__(self, validation_accuracy):
        if self.best_acc is None:
            self.best_acc = validation_accuracy
            return False
        elif validation_accuracy > self.best_acc + self.min_delta:
            self.best_acc = validation_accuracy
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(train_set.num_classes, train_set.seq_len, train_set.num_channels)

    lr_ratios = [1, 0.5]
    hid_dims = [64, 128]
    n_layers = [3, 5]
    p_drops = [0.1, 0.3]
    combinations = list(itertools.product(lr_ratios, hid_dims, n_layers, p_drops))
    for combination in combinations:
        lr_ratio, hid_dim, n_layer, p_drop = combination[0], combination[1], combination[2], combination[3]
        print("lr_ratio="+str(lr_ratio))
        print("hid_dim="+str(hid_dim))
        print("n_layer="+str(n_layer))
        print("p_drop="+str(p_drop))
        model_name = "_" + str(lr_ratio) + "_" + str(hid_dim) + "_" + str(n_layer) + "_" + str(p_drop)

        # ------------------
        #       Model
        # ------------------
        model = BasicConvClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels, hid_dim, n_layer, p_drop
        ).to(args.device)

        # ------------------
        #     Optimizer
        # ------------------
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*lr_ratio, weight_decay=1e-5) 

        # ------------------
        #   Start training
        # ------------------  
        max_val_acc = 0
        accuracy = Accuracy(
            task="multiclass", num_classes=train_set.num_classes, top_k=10
        ).to(args.device)
    
        early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            
            train_loss, train_acc, val_loss, val_acc = [], [], [], []
            
            model.train()
            for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
                X, y = X.to(args.device), y.to(args.device)

                y_pred = model(X)
                
                loss = F.cross_entropy(y_pred, y)
                train_loss.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                acc = accuracy(y_pred, y)
                train_acc.append(acc.item())

            model.eval()
            for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
                X, y = X.to(args.device), y.to(args.device)
                
                with torch.no_grad():
                    y_pred = model(X)

                val_loss.append(F.cross_entropy(y_pred, y).item())
                val_acc.append(accuracy(y_pred, y).item())

            print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
            torch.save(model.state_dict(), os.path.join(logdir, "model_last"+model_name+".pt"))
            if args.use_wandb:
                wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
            
            if np.mean(val_acc) > max_val_acc:
                cprint("New best.", "cyan")
                torch.save(model.state_dict(), os.path.join(logdir, "model_best"+model_name+".pt"))
                max_val_acc = np.mean(val_acc)

            if early_stopping(np.mean(val_acc)):
                print("Early stopping")
                break
                
        
        # ----------------------------------
        #  Start evaluation with best model
        # ----------------------------------
        model.load_state_dict(torch.load(os.path.join(logdir, "model_best"+model_name+".pt"), map_location=args.device))

        preds = [] 
        model.eval()
        for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
            preds.append(model(X.to(args.device)).detach().cpu())
            
        preds = torch.cat(preds, dim=0).numpy()
        np.save(os.path.join(logdir, "submission"+model_name), preds)
        cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()