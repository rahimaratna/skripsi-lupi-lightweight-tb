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


class TBMultimodalDataset(Dataset):
    """
    Dataset serbaguna untuk mode Unimodal (citra saja) maupun Multimodal LUPI (citra + teks).
    Kolom wajib pada CSV:
      - 'image_path' (atau 'filepath')
      - 'label' (0 = Normal, 1 = TB, atau string 'normal'/'tb')
    Kolom opsional (wajib saat mode='lupi' pada split train):
      - 'report' (atau 'text' / 'findings')
    """
    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose,
        tokenizer=None,
        max_len: int = 256,
        require_text: bool = False,
        root_dir: str = "",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.require_text = require_text
        self.root_dir = root_dir

        # Deteksi nama kolom path citra
        for col in ["image_path", "filepath", "path", "filename"]:
            if col in self.df.columns:
                self.img_col = col
                break
        else:
            raise ValueError(f"Kolom path citra tidak ditemukan. Kolom tersedia: {list(self.df.columns)}")

        # Deteksi nama kolom teks klinis
        self.txt_col = None
        for col in ["report", "text", "findings", "clinical_report"]:
            if col in self.df.columns:
                self.txt_col = col
                break

        if self.require_text and self.txt_col is None:
            raise ValueError("Mode LUPI membutuhkan kolom teks laporan ('report'/'text') pada data train.")

        # Normalisasi label ke integer {0, 1}
        if self.df["label"].dtype == object:
            label_map = {"normal": 0, "no_tb": 0, "0": 0, "tb": 1, "tuberculosis": 1, "1": 1}
            self.labels = self.df["label"].astype(str).str.lower().str.strip().map(label_map).astype(int).tolist()
        else:
            self.labels = self.df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = str(row[self.img_col])
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
            raw_text = str(row[self.txt_col]) if pd.notna(row[self.txt_col]) else "No clinical findings."
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
) -> DataLoader:
    df = pd.read_csv(csv_path)
    train_tf, eval_tf = get_transforms(img_size=img_size)
    tf = train_tf if is_train else eval_tf

    ds = TBMultimodalDataset(
        df=df,
        transform=tf,
        tokenizer=tokenizer,
        max_len=max_len,
        require_text=require_text,
        root_dir=root_dir,
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
