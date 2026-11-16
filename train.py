import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from dataset import build_tokenizer, create_dataloader
from models import TBClassifierLUPI
from losses import TBMultimodalLoss


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.0

    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    all_labels = []
    all_probs = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].numpy()

        out = model(image=images)
        probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()

        all_labels.append(labels)
        all_probs.append(probs)

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    return compute_metrics(y_true, y_prob)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    mode: str,
    grad_clip: float = 1.0,
) -> dict:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_align = 0.0
    steps = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        input_ids = batch["input_ids"].to(device, non_blocking=True) if mode == "lupi" else None
        attention_mask = batch["attention_mask"].to(device, non_blocking=True) if mode == "lupi" else None

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(image=images, input_ids=input_ids, attention_mask=attention_mask)
            loss_dict = criterion(
                logits=out["logits"],
                labels=labels,
                z_img=out.get("z_img"),
                z_txt=out.get("z_txt"),
            )
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_ce += float(loss_dict["loss_ce"].item())
        total_align += float(loss_dict["loss_align"].item())
        steps += 1

    return {
        "train_loss": total_loss / max(steps, 1),
        "train_ce": total_ce / max(steps, 1),
        "train_align": total_align / max(steps, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Eksperimen TB Classification: Unimodal vs Multimodal LUPI")
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["vgg11", "mobilenetv3", "shufflenetv2", "efficientnet_b0", "starnet", "vit"],
                        help="Pilihan vision encoder M1-M5")
    parser.add_argument("--mode", type=str, required=True, choices=["unimodal", "lupi"],
                        help="Skema pelatihan: unimodal atau lupi")

    parser.add_argument("--train_csv", type=str, required=True, help="CSV split train Shenzhen")
    parser.add_argument("--val_csv", type=str, required=True, help="CSV split validation Shenzhen")
    parser.add_argument("--test_shenzhen_csv", type=str, required=True, help="CSV split test Shenzhen")
    parser.add_argument("--test_montgomery_csv", type=str, default="", help="CSV uji eksternal Montgomery")
    parser.add_argument("--test_tbx11k_csv", type=str, default="", help="CSV uji eksternal TBX11K")
    parser.add_argument("--root_dir", type=str, default="", help="Prefix folder gambar jika path di CSV relatif")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_align", type=float, default=0.5, help="Bobot alignment loss untuk mode LUPI")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature SupCon loss")
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience berdasarkan Val MCC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = f"{args.backbone}_{args.mode}_seed{args.seed}"
    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Memulai Eksperimen: {exp_name} | Device: {device} ===")

    tokenizer = build_tokenizer() if args.mode == "lupi" else None

    train_loader = create_dataloader(
        csv_path=args.train_csv,
        batch_size=args.batch_size,
        is_train=True,
        require_text=(args.mode == "lupi"),
        tokenizer=tokenizer,
        use_balanced_sampler=True,
        num_workers=args.num_workers,
        root_dir=args.root_dir,
    )
    val_loader = create_dataloader(
        csv_path=args.val_csv,
        batch_size=args.batch_size,
        is_train=False,
        require_text=False,
        num_workers=args.num_workers,
        root_dir=args.root_dir,
    )

    model = TBClassifierLUPI(
        backbone_name=args.backbone,
        mode=args.mode,
        proj_dim=args.proj_dim,
        pretrained=True,
    ).to(device)

    criterion = TBMultimodalLoss(
        mode=args.mode,
        lambda_align=args.lambda_align,
        temperature=args.temperature,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_mcc = -1.0
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(save_dir, f"best_{exp_name}.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mode=args.mode,
        )
        val_stats = evaluate(model, val_loader, device)
        scheduler.step()

        row = {"epoch": epoch, **tr_stats, **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(row)

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] "
            f"Loss: {tr_stats['train_loss']:.4f} | "
            f"Val MCC: {val_stats['mcc']:.4f} | "
            f"Val BAcc: {val_stats['balanced_acc']:.4f} | "
            f"Val AUC: {val_stats['roc_auc']:.4f}"
        )

        if val_stats["mcc"] > best_val_mcc:
            best_val_mcc = val_stats["mcc"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mcc": best_val_mcc,
                "args": vars(args),
            }, best_ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping terpicu pada epoch {epoch} (Best Val MCC: {best_val_mcc:.4f})")
                break

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    print("
=== Memuat Checkpoint Terbaik untuk Evaluasi Test & Eksternal ===")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_targets = {"shenzhen_test": args.test_shenzhen_csv}
    if args.test_montgomery_csv:
        eval_targets["montgomery_ext"] = args.test_montgomery_csv
    if args.test_tbx11k_csv:
        eval_targets["tbx11k_ext"] = args.test_tbx11k_csv

    final_results = {
        "experiment": exp_name,
        "backbone": args.backbone,
        "mode": args.mode,
        "best_epoch": ckpt["epoch"],
        "best_val_mcc": best_val_mcc,
        "datasets": {},
    }

    for ds_name, csv_file in eval_targets.items():
        test_loader = create_dataloader(
            csv_path=csv_file,
            batch_size=args.batch_size,
            is_train=False,
            require_text=False,
            num_workers=args.num_workers,
            root_dir=args.root_dir,
        )
        metrics = evaluate(model, test_loader, device)
        final_results["datasets"][ds_name] = metrics
        print(
            f"[{ds_name.upper()}] MCC: {metrics['mcc']:.4f} | "
            f"BAcc: {metrics['balanced_acc']:.4f} | "
            f"AUC: {metrics['roc_auc']:.4f} | "
            f"Sens: {metrics['sensitivity']:.4f} | "
            f"Spec: {metrics['specificity']:.4f}"
        )

    with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)


if __name__ == "__main__":
    main()
