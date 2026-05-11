"""
天气分类模型训练主程序
策略: 两阶段训练 + MixUp + LabelSmoothing + CosineAnnealing
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from config import (
    MODEL_NAME, NUM_CLASSES, NUM_EPOCHS, WARMUP_EPOCHS,
    LEARNING_RATE, FINE_TUNE_LR, WEIGHT_DECAY, OPTIMIZER,
    LR_SCHEDULER, LR_MIN, WARMUP_EPOCHS_SCHEDULER,
    LABEL_SMOOTHING, MIXUP_ALPHA, CUTMIX_ALPHA, GRAD_CLIP,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_METRIC,
    CKPT_DIR, DEVICE, USE_AMP,
)
from model import (
    get_model, freeze_backbone, unfreeze_all,
    get_parameter_groups, count_parameters,
)
from dataset import create_dataloaders, mixup_data, cutmix_data
from utils import (
    AverageMeter, compute_metrics, EarlyStopping,
    TrainingLogger, set_seed, save_checkpoint, get_lr,
)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, amp_enabled):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    all_preds, all_targets = [], []

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        # ---- MixUp / CutMix ----
        use_mixup = MIXUP_ALPHA > 0 and epoch >= WARMUP_EPOCHS
        use_cutmix = CUTMIX_ALPHA > 0 and epoch >= WARMUP_EPOCHS

        if use_mixup and torch.rand(1).item() < 0.5:
            images, targets_a, targets_b, lam = mixup_data(images, targets, MIXUP_ALPHA)
            mixup_active = True
        elif use_cutmix and torch.rand(1).item() < 0.5:
            images, targets_a, targets_b, lam = cutmix_data(images, targets, CUTMIX_ALPHA)
            mixup_active = True
        else:
            mixup_active = False

        # ---- Forward ----
        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            with autocast():
                outputs = model(images)
                if mixup_active:
                    loss = lam * criterion(outputs, targets_a) + \
                           (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if mixup_active:
                loss = lam * criterion(outputs, targets_a) + \
                       (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # ---- 记录 ----
        losses.update(loss.item(), images.size(0))
        if not mixup_active:
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    metrics = {"loss": losses.avg}
    if all_preds:
        metrics.update(compute_metrics(all_preds, all_targets))

    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion):
    """验证"""
    model.eval()
    losses = AverageMeter()
    all_preds, all_targets = [], []

    for images, targets in dataloader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), images.size(0))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    metrics = {"loss": losses.avg}
    metrics.update(compute_metrics(all_preds, all_targets))

    return metrics


def build_optimizer(model, lr=LEARNING_RATE):
    """构建优化器"""
    if OPTIMIZER == "adamw":
        return optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER == "sgd":
        return optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")


def build_scheduler(optimizer, num_epochs, steps_per_epoch):
    """构建学习率调度器"""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS_SCHEDULER * steps_per_epoch

    if LR_SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=LR_MIN,
        )
    elif LR_SCHEDULER == "cosine_warm":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=total_steps // 3, T_mult=2, eta_min=LR_MIN,
        )
    elif LR_SCHEDULER == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs // 3, gamma=0.1,
        )
    else:
        scheduler = None

    # WarmUp
    if warmup_steps > 0 and scheduler is not None:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, scheduler],
            milestones=[warmup_steps],
        )
    elif warmup_steps > 0:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps,
        )

    return scheduler


def train():
    """主训练流程"""
    # ---- 初始化 ----
    set_seed(42)
    print(f"Device: {DEVICE}")
    print(f"AMP enabled: {USE_AMP}")

    # ---- 数据 ----
    dataloaders = create_dataloaders()
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")

    if train_loader is None:
        raise FileNotFoundError(
            "训练数据未找到! 请按以下结构放置数据:\n"
            "  data/train/sunny/*.jpg\n"
            "  data/train/rainy/*.jpg\n"
            "  data/train/cloudy/*.jpg\n"
            "  data/train/snowy/*.jpg\n"
            "  -- 或者 --\n"
            "  data/train/*.jpg + data/train_labels.csv"
        )

    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader) if val_loader else 0}")

    # ---- 模型 ----
    model = get_model(MODEL_NAME, pretrained=True)
    model = model.to(DEVICE)

    total_params, trainable_params = count_parameters(model)
    print(f"Model: {MODEL_NAME} | "
          f"Total params: {total_params/1e6:.2f}M | "
          f"Trainable: {trainable_params/1e6:.2f}M")

    # ---- 损失函数 ----
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ---- 第一阶段: WarmUp (仅训练分类头) ----
    if WARMUP_EPOCHS > 0:
        print("\n" + "="*60)
        print(f"Phase 1: WarmUp - 仅训练分类头 ({WARMUP_EPOCHS} epochs)")
        print("="*60)

        model = freeze_backbone(model)
        optimizer = build_optimizer(model, lr=LEARNING_RATE)
        scaler = GradScaler(enabled=USE_AMP)

        for epoch in range(WARMUP_EPOCHS):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                epoch, USE_AMP,
            )
            val_metrics = validate(model, val_loader, criterion) if val_loader else {}
            lr = get_lr(optimizer)
            print(f"  WarmUp {epoch+1}/{WARMUP_EPOCHS} | "
                  f"loss={train_metrics['loss']:.4f} "
                  f"val_acc={val_metrics.get('accuracy', 0):.4f} "
                  f"val_f1={val_metrics.get('f1_macro', 0):.4f}")

    # ---- 第二阶段: 全模型微调 ----
    print("\n" + "="*60)
    print(f"Phase 2: Fine-Tune - 全模型训练 "
          f"({NUM_EPOCHS - WARMUP_EPOCHS} epochs)")
    print("="*60)

    model = unfreeze_all(model)
    param_groups = get_parameter_groups(model, FINE_TUNE_LR, LEARNING_RATE)
    optimizer = build_optimizer(model, lr=FINE_TUNE_LR)

    # 用分组学习率重建优化器
    if OPTIMIZER == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(param_groups, momentum=0.9,
                              weight_decay=WEIGHT_DECAY, nesterov=True)

    scheduler = build_scheduler(
        optimizer, NUM_EPOCHS, len(train_loader),
    )
    scaler = GradScaler(enabled=USE_AMP)
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, mode="max",
    )
    logger = TrainingLogger()
    best_f1 = 0.0
    best_ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")

    for epoch in range(WARMUP_EPOCHS, NUM_EPOCHS):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, USE_AMP,
        )

        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, criterion)
        else:
            val_metrics = {}

        # LR step
        current_lr = get_lr(optimizer)
        if scheduler:
            scheduler.step()

        # Log
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

        # Save best
        metric_key = EARLY_STOPPING_METRIC if EARLY_STOPPING_METRIC in val_metrics else "loss"
        current_score = val_metrics.get(metric_key, train_metrics.get("loss", 999))
        if metric_key != "loss":
            current_score = -current_score if metric_key == "loss" else current_score

        if current_score > best_f1:
            best_f1 = current_score
            save_checkpoint(
                model, optimizer, epoch, val_metrics, best_ckpt_path,
                scheduler=scheduler,
            )
            print(f"  => Best model saved (val_{metric_key}={val_metrics.get(metric_key, 0):.4f})")

        # Early stopping
        if early_stopping(val_metrics.get(metric_key, 0), epoch):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    # ---- 训练完成 ----
    logger.save()
    print(f"\nTraining complete! Best model saved to: {best_ckpt_path}")

    # 清理GPU内存
    del scaler, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, best_ckpt_path


if __name__ == "__main__":
    train()
