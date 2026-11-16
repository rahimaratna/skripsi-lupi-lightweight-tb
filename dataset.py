import os
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(img_size: int = 224) -> Tuple[A.Compose, A.Compose]:
    train_tf = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-7, 7), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    eval_tf = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, eval_tf


class ImbalancedDatasetSampler(Sampler):
    """Oversampling berbasis invers frekuensi kelas agar batch training seimbang."""
    def __init__(self, labels: list):
        self.indices = list(range(len(labels)))
        self.num_samples = len(self.indices)

        label_to_count: Dict[int, int] = {}
        for idx in self.indices:
            lbl = int(labels[idx])
            label_to_count[lbl] = label_to_count.get(lbl, 0) + 1

        weights = [1.0 / label_to_count[int(labels[idx])] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


def load_split_dataframe(csv_path: str) -> pd.DataFrame:
    """Membaca CSV baik yang memiliki header maupun yang tanpa header (kolom 0=filename, kolom 1=label)."""
    df_peek = pd.read_csv(csv_path, nrows=1, header=None)
    first_val = str(df_peek.iloc[0, 0]).strip().lower()
    known_headers = {"image_path", "filepath", "path", "filename", "img", "image"}

    if first_val in known_headers:
        return pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] == 2:
            df.columns = ["image_path", "label"]
        elif df.shape[1] >= 3:
            df.columns = ["image_path", "label", "report"] + [f"extra_{i}" for i in range(df.shape[1] - 3)]
        else:
            raise ValueError(f"Format CSV tidak dikenali ({df.shape[1]} kolom) pada {csv_path}")
        return df


class TBMultimodalDataset(Dataset):
    """
    Dataset untuk mode Unimodal (citra saja) dan Multimodal LUPI (citra + laporan .txt).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose,
        tokenizer=None,
        max_len: int = 256,
        require_text: bool = False,
        root_dir: str = "",
        reports_dir: str = "",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.require_text = require_text
        self.root_dir = root_dir
        self.reports_dir = reports_dir

        for col in ["image_path", "filepath", "path", "filename", "img"]:
            if col in self.df.columns:
                self.img_col = col
                break
        else:
            raise ValueError(f"Kolom path citra tidak ditemukan. Kolom tersedia: {list(self.df.columns)}")

        self.txt_col = None
        for col in ["report", "text", "findings", "clinical_report"]:
            if col in self.df.columns:
                self.txt_col = col
                break

        if self.require_text and (self.txt_col is None) and (not self.reports_dir or not os.path.isdir(self.reports_dir)):
            raise ValueError(
                "Mode LUPI membutuhkan kolom teks pada CSV atau folder --reports_dir yang valid."
            )

        if self.df["label"].dtype == object:
            label_map = {"normal": 0, "no_tb": 0, "0": 0, "tb": 1, "tuberculosis": 1, "1": 1}
            self.labels = self.df["label"].astype(str).str.lower().str.strip().map(label_map).astype(int).tolist()
        else:
            self.labels = self.df["label"].astype(int).tolist()

    def _read_report_text(self, row: pd.Series, raw_img_name: str) -> str:
        # 1. Jika ada kolom teks langsung di CSV
        if self.txt_col is not None and pd.notna(row[self.txt_col]):
            val = str(row[self.txt_col]).strip()
            if val.endswith(".txt") and self.reports_dir:
                txt_path = os.path.join(self.reports_dir, os.path.basename(val))
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read().strip()
            return val

        # 2. Cari file .txt di folder reports_dir berdasarkan nama file citra (misal CHNCXR_0278_0.png -> CHNCXR_0278_0.txt)
        if self.reports_dir:
            stem = os.path.splitext(os.path.basename(raw_img_name))[0]
            txt_path = os.path.join(self.reports_dir, f"{stem}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if content:
                        return content

        return "No clinical findings."

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        raw_img_name = str(row[self.img_col]).strip()
        img_path = raw_img_name
        if self.root_dir and not os.path.isabs(img_path):
            img_path = os.path.join(self.root_dir, img_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Gagal membaca gambar: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(image=img)["image"]

        item = {
            "image": img_tensor,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "image_path": img_path,
        }

        if self.require_text and self.tokenizer is not None:
            raw_text = self._read_report_text(row, raw_img_name)
            enc = self.tokenizer(
                raw_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            item["input_ids"] = enc["input_ids"].squeeze(0)
            item["attention_mask"] = enc["attention_mask"].squeeze(0)

        return item


def build_tokenizer(model_name: str = "microsoft/BiomedVLP-CXR-BERT-general"):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def create_dataloader(
    csv_path: str,
    batch_size: int = 16,
    img_size: int = 224,
    is_train: bool = False,
    require_text: bool = False,
    tokenizer=None,
    max_len: int = 256,
    use_balanced_sampler: bool = True,
    num_workers: int = 2,
    root_dir: str = "",
    reports_dir: str = "",
) -> DataLoader:
    df = load_split_dataframe(csv_path)
    train_tf, eval_tf = get_transforms(img_size=img_size)
    tf = train_tf if is_train else eval_tf

    ds = TBMultimodalDataset(
        df=df,
        transform=tf,
        tokenizer=tokenizer,
        max_len=max_len,
        require_text=require_text,
        root_dir=root_dir,
        reports_dir=reports_dir,
    )

    sampler = ImbalancedDatasetSampler(ds.labels) if (is_train and use_balanced_sampler) else None
    shuffle = is_train and (sampler is None)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
