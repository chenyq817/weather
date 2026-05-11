"""
工具函数: 指标计算, 日志, 可视化
"""

import os
import time
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

import torch
from config import CLASS_NAMES, LOG_DIR


class AverageMeter:
    """追踪和计算平均值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(predictions, targets):
    """计算分类指标"""
    preds = np.array(predictions)
    trues = np.array(targets)

    metrics = {
        "accuracy": accuracy_score(trues, preds),
        "f1_macro": f1_score(trues, preds, average="macro"),
        "f1_weighted": f1_score(trues, preds, average="weighted"),
        "precision_macro": precision_score(trues, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(trues, preds, average="macro", zero_division=0),
    }

    # 每个类别的F1
    per_class_f1 = f1_score(trues, preds, average=None, zero_division=0)
    for i, cls_name in enumerate(CLASS_NAMES):
        if i < len(per_class_f1):
            metrics[f"f1_{cls_name}"] = per_class_f1[i]

    return metrics


def compute_confusion_matrix(predictions, targets):
    """计算混淆矩阵"""
    return confusion_matrix(targets, predictions).tolist()


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir=LOG_DIR):
        self.log_dir = log_dir
        self.metrics_history = {
            "train": [],
            "val": [],
        }
        self.start_time = time.time()

    def log_epoch(self, epoch, train_metrics, val_metrics, lr):
        """记录每个epoch的指标"""
        entry = {
            "epoch": epoch + 1,
            "time_elapsed": time.time() - self.start_time,
            "lr": lr,
            "train": train_metrics,
            "val": val_metrics,
        }
        self.metrics_history["train"].append({"epoch": epoch + 1, **train_metrics})
        self.metrics_history["val"].append({"epoch": epoch + 1, **val_metrics})

        # 打印摘要
        train_str = f"train_loss={train_metrics.get('loss', 0):.4f} train_acc={train_metrics.get('accuracy', 0):.4f}"
        val_str = f"val_loss={val_metrics.get('loss', 0):.4f} val_f1={val_metrics.get('f1_macro', 0):.4f} val_acc={val_metrics.get('accuracy', 0):.4f}"
        print(f"Epoch {epoch+1:3d} | {train_str} | {val_str} | lr={lr:.2e}")

    def save(self, filepath=None):
        """保存训练历史到JSON"""
        if filepath is None:
            filepath = os.path.join(self.log_dir, "training_history.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)

    def get_best_epoch(self, metric="f1_macro"):
        """获取验证集上最佳epoch"""
        best_idx = 0
        best_val = -float("inf")
        for i, entry in enumerate(self.metrics_history["val"]):
            val = entry.get(metric, 0)
            if val > best_val:
                best_val = val
                best_idx = i
        return best_idx, best_val


def set_seed(seed=42):
    """固定随机种子确保可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(model, optimizer, epoch, metrics, filepath, scheduler=None):
    """保存检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, scheduler=None):
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
