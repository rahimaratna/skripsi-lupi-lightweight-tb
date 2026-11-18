import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)

from dataset import TBMultimodalDataset, get_transforms, read_split_csv
from models import (
    TBClassifierUnimodal,
    TBClassifierLUPI,
    measure_inference_efficiency,
)
from losses import LUPITotalLoss


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_splits(args):
    if args.train_csv and args.val_csv:
        train_df = read_split_csv(args.train_csv)
        val_df = read_split_csv(args.val_csv)
        test_df = (
            read_split_csv(args.test_csv)
            if args.test_csv and os.path.exists(args.test_csv)
            else None
        )
        return train_df, val_df, test_df

    if args.csv_path and os.path.exists(args.csv_path):
        df = read_split_csv(args.csv_path)
        split_col = next(
            (c for c in ["split", "subset", "fold_split"] if c in df.columns), None
        )
        if split_col is None:
            raise ValueError("CSV tunggal harus memiliki kolom 'split' (train/val/test).")
        train_df = df[df[split_col].str.lower() == "train"].reset_index(drop=True)
        val_df = df[
            df[split_col].str.lower().isin(["val", "valid", "validation"])
        ].reset_index(drop=True)
        test_df = df[df[split_col].str.lower() == "test"].reset_index(drop=True)
        return train_df, val_df, test_df if len(test_df) > 0 else None

    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "val.csv")
    test_path = os.path.join(args.data_dir, "test.csv")
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_df = read_split_csv(train_path)
        val_df = read_split_csv(val_path)
        test_df = read_split_csv(test_path) if os.path.exists(test_path) else None
        return train_df, val_df, test_df

    raise FileNotFoundError(
        "Tidak ditemukan file CSV dataset. Berikan --csv_path atau --train_csv & --val_csv."
    )


def build_inverse_freq_sampler(dataset: TBMultimodalDataset, seed: int) -> WeightedRandomSampler:
    """
    Inverse-frequency weighted sampling dengan replacement (Rajaraman et al., 2026, hlm. 3).
    """
    labels = [sample["label"] for sample in dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.DoubleTensor([class_weights[y] for y in labels])

    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def compute_metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    bal_acc = 0.5 * (sens + spec)
    youden_j = sens + spec - 1.0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.0

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "bal_acc": float(bal_acc),
        "sens": float(sens),
        "spec": float(spec),
        "prec": float(prec),
        "npv": float(npv),
        "f1": float(f1),
        "mcc": float(mcc),
        "j": float(youden_j),
        "auroc": float(auroc),
        "auc": float(auroc),
        "acc": float(acc),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_labels = []
    all_probs = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=-1)[:, 1]

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.50):
    y_true, y_prob = predict_probs(model, loader, device)
    return compute_metrics_from_probs(y_true, y_prob, threshold=threshold)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Mencari threshold optimal t* pada validation set untuk memaksimalkan MCC
    (tanpa menyentuh test set sehingga bebas data leakage).
    """
    best_thresh = 0.50
    best_mcc = matthews_corrcoef(y_true, (y_prob >= 0.50).astype(int))

    for thresh in np.linspace(0.10, 0.90, 161):
        preds = (y_prob >= thresh).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc + 1e-6:
            best_mcc = mcc
            best_thresh = float(thresh)

    return round(best_thresh, 4)


def main():
    parser = argparse.ArgumentParser(
        description="Training Unimodal vs LUPI TB Classification (M1-M8)"
    )
    parser.add_argument("--model_code", type=str, default="M1-L")
    parser.add_argument("--mode", type=str, default="lupi", choices=["unimodal", "lupi"])
    parser.add_argument("--backbone", type=str, default="vgg11_bn")
    parser.add_argument(
        "--text_model", type=str, default="microsoft/BiomedVLP-CXR-BERT-general"
    )
    parser.add_argument("--text_freeze_policy", type=str, default="authors_default")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--report_dir", type=str, default="reports")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_align", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument(
        "--test_csvs",
        nargs="*",
        default=[],
        help="Pasangan NamaDataset=PathCSV, contoh: Shenzhen=/path/label_test.csv",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(
        f"[{args.model_code}] Mode: {args.mode} | Backbone: {args.backbone} | Seed: {args.seed}"
    )
    if args.mode == "lupi":
        print(
            f"LUPI PARAMS | report_dir={args.report_dir} | lambda={args.lambda_align} | "
            f"tau={args.temperature} | freeze={args.text_freeze_policy}"
        )
    print("=" * 75)

    train_df, val_df, test_df = load_splits(args)
    tokenizer = (
        AutoTokenizer.from_pretrained(args.text_model) if args.mode == "lupi" else None
    )

    train_dataset = TBMultimodalDataset(
        df=train_df,
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        mode=args.mode,
        tokenizer=tokenizer,
        max_len=args.max_len,
        transform=get_transforms(args.img_size, is_train=True),
        require_report=(args.mode == "lupi"),
    )
    val_dataset = TBMultimodalDataset(
        df=val_df,
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        mode="unimodal",
        tokenizer=None,
        transform=get_transforms(args.img_size, is_train=False),
        require_report=False,
    )

    train_sampler = build_inverse_freq_sampler(train_dataset, seed=args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(args.mode == "lupi"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.mode == "unimodal":
        model = TBClassifierUnimodal(
            backbone_name=args.backbone,
            proj_dim=args.proj_dim,
            num_classes=2,
            pretrained=True,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = TBClassifierLUPI(
            backbone_name=args.backbone,
            text_model_name=args.text_model,
            proj_dim=args.proj_dim,
            num_classes=2,
            pretrained=True,
            text_freeze_policy=args.text_freeze_policy,
        ).to(device)
        criterion = LUPITotalLoss(
            lambda_align=args.lambda_align,
            temperature=args.temperature,
        )

    efficiency = measure_inference_efficiency(model, device, img_size=args.img_size)
    print(
        f"Efficiency ({args.model_code}) -> Infer Params: {efficiency['infer_params_m']}M | "
        f"Size: {efficiency['infer_size_mb']}MB | Latency: {efficiency['latency_ms']}ms"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_mcc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_cls_img = 0.0
        running_cls_txt = 0.0
        running_ntxent = 0.0
        running_cos = 0.0
        steps = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            if args.mode == "unimodal":
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_cls_img += loss.item()
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                logits_img, logits_txt, z_img, z_txt = model(
                    images, input_ids=input_ids, attention_mask=attention_mask
                )
                loss, loss_parts = criterion(
                    logits_img, logits_txt, z_img, z_txt, labels
                )
                loss.backward()
                optimizer.step()

                running_loss += loss_parts["total"]
                running_cls_img += loss_parts["cls_img"]
                running_cls_txt += loss_parts["cls_txt"]
                running_ntxent += loss_parts["ntxent"]
                running_cos += loss_parts["cos"]

            steps += 1

        avg_loss = running_loss / max(steps, 1)
        val_metrics = evaluate(model, val_loader, device, threshold=0.50)
        val_mcc = val_metrics["mcc"]

        row_log = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "cls_img": round(running_cls_img / max(steps, 1), 4),
            "cls_txt": round(running_cls_txt / max(steps, 1), 4),
            "ntxent": round(running_ntxent / max(steps, 1), 4),
            "cos": round(running_cos / max(steps, 1), 4),
            "val_auc": round(val_metrics["auc"], 4),
            "val_mcc": round(val_mcc, 4),
            "val_sens": round(val_metrics["sens"], 4),
            "val_spec": round(val_metrics["spec"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_acc": round(val_metrics["acc"], 4),
        }
        history.append(row_log)

        if args.mode == "lupi":
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} "
                f"(cls_i={row_log['cls_img']:.3f}, cls_t={row_log['cls_txt']:.3f}, "
                f"ntx={row_log['ntxent']:.3f}, cos={row_log['cos']:.3f}) | "
                f"Val MCC: {val_mcc:.4f} | AUC: {val_metrics['auc']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} | "
                f"Val MCC: {val_mcc:.4f} | AUC: {val_metrics['auc']:.4f}"
            )

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping di epoch {epoch} (best Val MCC: {best_val_mcc:.4f} @ ep {best_epoch})"
                )
                break

    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)

    # Muat checkpoint terbaik & kalibrasi threshold optimal t* pada validation set
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_y_true, val_y_prob = predict_probs(model, val_loader, device)
    optimal_threshold = find_optimal_threshold(val_y_true, val_y_prob)
    val_calibrated = compute_metrics_from_probs(val_y_true, val_y_prob, threshold=optimal_threshold)
    print(
        f"Kalibrasi Validation Set -> Optimal Threshold t* = {optimal_threshold:.4f} "
        f"(Val MCC: {best_val_mcc:.4f} -> {val_calibrated['mcc']:.4f})"
    )

    summary = {
        "model_code": args.model_code,
        "config": vars(args),
        "efficiency": efficiency,
        "best_epoch": best_epoch,
        "best_val_mcc": best_val_mcc,
        "optimal_threshold": optimal_threshold,
    }

    test_targets = []
    if args.test_csvs:
        for item in args.test_csvs:
            if "=" in item:
                ds_name, ds_path = item.split("=", 1)
                if os.path.exists(ds_path):
                    test_targets.append((ds_name, ds_path))
    elif test_df is not None and len(test_df) > 0:
        test_targets.append(("Default", args.test_csv))

    if len(test_targets) > 0:
        summary["test_results"] = {}
        summary["test_results_thresh50"] = {}

        print("-" * 75)
        for ds_name, ds_path in test_targets:
            cur_test_df = read_split_csv(ds_path) if ds_path else test_df
            test_dataset = TBMultimodalDataset(
                df=cur_test_df,
                data_dir=args.data_dir,
                report_dir=args.report_dir,
                mode="unimodal",
                tokenizer=None,
                transform=get_transforms(args.img_size, is_train=False),
                require_report=False,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            y_true_t, y_prob_t = predict_probs(model, test_loader, device)
            metrics_cal = compute_metrics_from_probs(y_true_t, y_prob_t, threshold=optimal_threshold)
            metrics_50 = compute_metrics_from_probs(y_true_t, y_prob_t, threshold=0.50)

            summary["test_results"][ds_name] = metrics_cal
            summary["test_results_thresh50"][ds_name] = metrics_50
            print(
                f"TEST [{args.model_code} | {ds_name} | t*={optimal_threshold:.2f}] -> "
                f"AUROC: {metrics_cal['auroc']:.4f} | MCC: {metrics_cal['mcc']:.4f} "
                f"(t=0.50: {metrics_50['mcc']:.4f}) | Sens: {metrics_cal['sens']:.4f} | Spec: {metrics_cal['spec']:.4f}"
            )
        print("-" * 75)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()