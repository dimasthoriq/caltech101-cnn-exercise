"""
Author: Dimas Ahmad
Description: This file contains the trainer functions for the project
"""

import os
import torch
import time
import datetime
import torcheval.metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, best_loss, current_loss):
        if current_loss <= (best_loss - self.min_delta):
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer

        self.best_loss = float('inf')
        self.best_acc = 0
        self.best_epoch = 0
        self.best_model = self.model.state_dict()

        self.early_stopping = EarlyStopping(config['patience'], config['min_delta'])
        self.device = config['device']
        self.max_epochs = config['max_epochs']
        self.print_freq = config['print_freq']

        self.train_acc = torcheval.metrics.MulticlassAccuracy(device=self.device)
        self.val_acc = torcheval.metrics.MulticlassAccuracy(device=self.device)
        self.config = config
        self.suffix = f"{config['archi']}arch_{config['normalization']}norm_" \
                      f"{config['dropout_rate']}do_{config['weight_decay']}wd_" \
                      f"{config['loss']}loss_{config['compression']}trunc_" \
                      f"{config['learning_rate']}lr_{config['batch_size']}batch_" \
                      f"{config['from_scratch']}scratch"

    def train_one_epoch(self):
        epoch_loss = 0
        self.model.train()
        self.train_acc.reset()

        for images, targets in self.train_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.detach().cpu().item()
            self.train_acc.update(outputs, targets)

        epoch_loss /= len(self.train_loader)
        epoch_acc = self.train_acc.compute().detach().cpu().item()

        return epoch_loss, epoch_acc

    def validate(self):
        epoch_loss = 0
        self.model.eval()
        self.val_acc.reset()

        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                epoch_loss += loss.detach().cpu().item()
                self.val_acc.update(outputs, targets)

        epoch_loss /= len(self.val_loader)
        epoch_acc = self.val_acc.compute().detach().cpu().item()

        return epoch_loss, epoch_acc

    def fit(self):
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        time_start = time.time()
        for epoch in range(self.max_epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            if epoch % self.print_freq == 0:
                print(f"Epoch {epoch+1}/{self.max_epochs+1} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Best Epoch: {self.best_epoch}")

            self.early_stopping(self.best_loss, val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping")
                print(f"Epoch {epoch + 1}/{self.max_epochs + 1} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Best Epoch: {self.best_epoch}")
                break

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_acc = val_acc
                self.best_epoch = epoch + 1
                self.best_model = self.model.state_dict()

        duration = time.time() - time_start
        self.model.load_state_dict(self.best_model)
        print('Training completed in {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        self.save_model()
        self.save_training_curves(train_losses[:self.best_epoch],
                                  val_losses[:self.best_epoch],
                                  train_accs[:self.best_epoch],
                                  val_accs[:self.best_epoch]
                                  )

    def save_model(self):
        time_stamp = datetime.datetime.today().strftime('%Y%m%d_%H%M')
        exp_path = "./Experiments/"
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        model_save_path = os.path.join(exp_path, f"model_{self.suffix}_{time_stamp}.pth")
        torch.save(self.model, model_save_path)

    def save_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        time_stamp = datetime.datetime.today().strftime('%Y%m%d_%H%M')
        exp_path = "./Experiments/"
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        curve_save_path = os.path.join(exp_path, f"curves_{self.suffix}_{time_stamp}.png")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Test')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train')
        plt.plot(range(1, len(val_accs) + 1), val_accs, label='Test')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(curve_save_path)
        plt.close()


def evaluate(model, data_loader, config):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(config['device']), targets.to(config['device'])
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    report = classification_report(all_targets, all_preds, digits=4, zero_division=0, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Save the report and matrix
    time_stamp = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    suffix = f"{config['archi']}arch_{config['normalization']}norm_"\
             f"{config['dropout_rate']}do_{config['weight_decay']}wd_"\
             f"{config['loss']}loss_{config['compression']}trunc_"\
             f"{config['learning_rate']}lr_{config['batch_size']}batch_"\
             f"{config['from_scratch']}scratch_{time_stamp}"

    exp_path = "./Experiments/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    report_save_path = os.path.join(exp_path, f"report_{suffix}.txt")
    cm_save_path = os.path.join(exp_path, f"cm_{suffix}.png")

    with open(report_save_path, 'w') as f:
        f.write(str(report))

    disp.plot()
    disp.figure_.savefig(cm_save_path)
