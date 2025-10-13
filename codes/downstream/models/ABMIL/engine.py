import os
import pickle
import numpy as np
from tqdm import tqdm

from typing import Dict
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.bootstrapping import BootStrapper
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification.f_beta import F1Score
from torchmetrics import AUROC
from torchmetrics.classification.confusion_matrix import ConfusionMatrix

import torch
import torch.nn.functional as F


class Engine(object):
    def __init__(self, args, results_dir, splits):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {split: {"Macro_AUC": 0.0, "Macro_ACC": 0.0, "Macro_F1": 0.0, "Weighted_AUC": 0.0, "Weighted_ACC": 0.0, "Weighted_F1": 0.0} for split in splits.keys()}
        self.results.pop("train")
        self.results_dir = results_dir
        self.filename_best = None
        self.best_epoch = 0
        self.epoch = 0
        self.early_stop = 10

    def learning(self, model, dataloaders, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        if self.args.resume is not None:
            print("=> loading checkpoint '{}'".format(self.args.resume))
            ckpt = torch.load(self.args.resume)
            if "results" in ckpt.keys():
                for split in self.results.keys():
                    if split in ckpt["results"].keys():
                        self.results[split] = ckpt["results"][split]
            print({k: round(v["Macro_AUC"], 4) for k, v in self.results.items()})
            self.epoch = self.best_epoch = ckpt["epoch"] if "epoch" in ckpt.keys() else ckpt["best_epoch"]
            model.load_state_dict(ckpt["state_dict"])
            print("=> loaded checkpoint (epoch {})".format(self.epoch))
        if self.args.evaluate:
            val_results, val_outputs = self.validate(dataloaders["val"], model, criterion, split="val")
            self.results["val"] = val_results
            with open(os.path.join(self.results_dir, "val_outputs.pkl"), "wb") as f:
                pickle.dump(val_outputs, f)
            for split in self.results.keys():
                if split in ["train", "val"]:
                    continue
                results, outputs = self.validate(dataloaders[split], model, criterion, split=split)
                self.results[split] = results
                with open(os.path.join(self.results_dir, "{}_outputs.pkl".format(split)), "wb") as f:
                    pickle.dump(outputs, f)
            return self.results

        for epoch in range(self.epoch, self.args.num_epoch):
            print("Epoch: {}".format(epoch))
            self.epoch = epoch
            self.train(dataloaders["train"], model, criterion, optimizer)
            # evaluate
            val_results, val_outputs = self.validate(dataloaders["val"], model, criterion, split="val")
            if val_results["Macro_AUC"] > self.results["val"]["Macro_AUC"]:
                self.results["val"] = val_results
                with open(os.path.join(self.results_dir, "val_outputs.pkl"), "wb") as f:
                    pickle.dump(val_outputs, f)
                for split in self.results.keys():
                    if split in ["train", "val"]:
                        continue
                    results, outputs = self.validate(dataloaders[split], model, criterion, split=split)
                    self.results[split] = results
                    with open(os.path.join(self.results_dir, "{}_outputs.pkl".format(split)), "wb") as f:
                        pickle.dump(outputs, f)
                #
                self.best_epoch = self.epoch
                self.save_checkpoint(
                    {
                        "epoch": self.best_epoch,
                        "state_dict": model.state_dict(),
                        "results": self.results,
                    }
                )
                self.early_stop = 0
            else:
                self.early_stop += 1

            if self.early_stop >= 10:
                print("Early stopping at epoch {}".format(self.epoch))
                break

            print(" *** best model {}".format(self.filename_best))
            for split in self.results.keys():
                if split in ["train"]:
                    continue
                print(" *** best Macro AUC results on {} split: {} at epoch {}".format(split, self.results[split]["Macro_AUC"], self.best_epoch))
            scheduler.step()
            print(">>>")
            print(">>>")
            print(">>>")
            print(">>>")
        return self.results

    def train(self, data_loader, model, criterion, optimizer):
        print("running train...")
        model.train()
        total_loss = 0.0
        all_logits = np.empty((0, self.args.num_classes))
        all_labels = np.empty((0, self.args.num_classes))
        dataloader = tqdm(data_loader, desc="train epoch {}".format(self.epoch)) if self.args.tqdm else data_loader
        for batch_idx, (data_Dataset, data_Case, data_Slide, data_Class, data_WSI, data_Coords, data_Label, data_Split) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device)
            data_Label = F.one_hot(data_Label, num_classes=self.args.num_classes).float().to(self.device)
            logit = model(data_WSI)
            loss = criterion(logit.view(1, -1), data_Label)
            # results
            all_labels = np.row_stack((all_labels, data_Label.cpu().numpy()))
            all_logits = np.row_stack((all_logits, torch.softmax(logit, dim=-1).detach().cpu().numpy()))
            total_loss += loss.item()
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # calculate loss
        loss = total_loss / len(dataloader)
        print("loss: {:.4f}".format(loss))
        # calculate metrics
        results = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
        #
        self.args.run.log(
            {
                "train_loss": loss,
                "train_Macro_AUC": results["Macro_AUC"],
                "train_Macro_ACC": results["Macro_ACC"],
                "train_Macro_F1": results["Macro_F1"],
                "train_Weighted_AUC": results["Weighted_AUC"],
                "train_Weighted_ACC": results["Weighted_ACC"],
                "train_Weighted_F1": results["Weighted_F1"],
            },
            step=self.epoch,
        )

    def validate(self, data_loader, model, criterion, split="val"):
        print("running {}...".format(split))
        model.eval()
        total_loss = 0.0
        all_ID = []
        all_Slide = []
        all_logits = np.empty((0, self.args.num_classes))
        all_labels = np.empty((0, self.args.num_classes))
        all_attns = []
        dataloader = tqdm(data_loader, desc="{} epoch {}".format(split, self.epoch)) if self.args.tqdm else data_loader
        for batch_idx, (data_Dataset, data_Case, data_Slide, data_Class, data_WSI, data_Coords, data_Label, data_Split) in enumerate(dataloader):
            all_ID.append(data_Case[0])
            all_Slide.append(data_Slide[0])
            data_WSI = data_WSI.to(self.device)
            data_Label = F.one_hot(data_Label, num_classes=self.args.num_classes).float().to(self.device)
            with torch.no_grad():
                logit, attn = model(data_WSI, return_attn=True)
                loss = criterion(logit.view(1, -1), data_Label)
            print(logit.shape, data_Label.shape, attn.shape)
            # results
            all_labels = np.row_stack((all_labels, data_Label.cpu().numpy()))
            all_logits = np.row_stack((all_logits, torch.softmax(logit, dim=-1).detach().cpu().numpy()))
            all_attns.append(attn.detach().cpu().numpy())
            total_loss += loss.item()
        # calculate loss
        loss = total_loss / len(dataloader)
        print("loss: {:.4f}".format(loss))

        results = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
        outputs = {"ID": all_ID, "slides": all_Slide, "logits": all_logits, "labels": all_labels, "attns": all_attns}
        #
        self.args.run.log(
            {
                "{}_loss".format(split): loss,
                "{}_Macro_AUC".format(split): results["Macro_AUC"],
                "{}_Macro_ACC".format(split): results["Macro_ACC"],
                "{}_Macro_F1".format(split): results["Macro_F1"],
                "{}_Weighted_AUC".format(split): results["Weighted_AUC"],
                "{}_Weighted_ACC".format(split): results["Weighted_ACC"],
                "{}_Weighted_F1".format(split): results["Weighted_F1"],
            },
            step=self.epoch,
        )
        return results, outputs

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir, "model_best_{epoch}.pth.tar".format(epoch=self.best_epoch))
        print("save best model {filename}".format(filename=self.filename_best))
        torch.save(state, self.filename_best)

    def metrics(self, logits, labels):
        meter = self.meter(num_classes=self.args.num_classes, bootstrap=False)
        results = meter(logits, labels)
        print("General Results:")
        print(
            "Macro AUC:    {:.4f},   Macro ACC:    {:.4f},   Macro F1:    {:.4f}".format(
                results["Macro_AUC"],
                results["Macro_ACC"],
                results["Macro_F1"],
            )
        )
        print(
            "Weighted AUC: {:.4f},   Weighted ACC: {:.4f},   Weighted F1: {:.4f}".format(
                results["Weighted_AUC"],
                results["Weighted_ACC"],
                results["Weighted_F1"],
            )
        )
        print("Confusion Matrix:")
        print(results["Confusion_Matrix"])
        results.pop("Confusion_Matrix")
        results = {k: round(v.item(), 4) for k, v in results.items()}
        return results

    def meter(self, num_classes, bootstrap=False):
        metrics: Dict[str, Metric] = {
            "Macro_ACC": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="macro").to(self.device),
            "Macro_F1": F1Score(num_classes=int(num_classes), average="macro", task="multiclass").to(self.device),
            "Macro_AUC": AUROC(num_classes=num_classes, average="macro", task="multiclass").to(self.device),
            "Weighted_ACC": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="weighted").to(self.device),
            "Weighted_F1": F1Score(num_classes=int(num_classes), average="weighted", task="multiclass").to(self.device),
            "Weighted_AUC": AUROC(num_classes=num_classes, average="weighted", task="multiclass").to(self.device),
            "Confusion_Matrix": ConfusionMatrix(num_classes=int(num_classes), task="multiclass").to(self.device),
        }
        # boot strap wrap
        if bootstrap:
            for k, m in metrics.items():
                # print("wrapping:", k)
                metrics[k] = BootStrapper(m, num_bootstraps=1000, sampling_strategy="multinomial").to(self.device)
        metrics = MetricCollection(metrics)
        return metrics
