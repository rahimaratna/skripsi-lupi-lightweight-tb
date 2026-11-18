import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size: int = 224, is_train: bool = True):
    """
    Pipeline augmentasi identik dengan paper rujukan:
    - Train: CLAHE (p=0.5) -> Resize (INTER_AREA) -> Normalize -> ToTensorV2
    - Val/Test: Resize (INTER_AREA) -> Normalize -> ToTensorV2
    Tanpa HorizontalFlip / Rotasi agar orientasi anatomi paru tetap selaras dengan laporan.
    """
    if is_train:
        return A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _detect_column(df: pd.DataFrame, candidates: list[str], col_desc: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Kolom {col_desc} tidak ditemukan. Kolom tersedia: {list(df.columns)}")


class TBMultimodalDataset(Dataset):
    """
    Dataset untuk mode 'unimodal' (citra saja) dan 'lupi' (citra + laporan klinis saat training).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        report_dir: str = "reports",
        mode: str = "lupi",
        tokenizer=None,
        max_len: int = 512,
        transform=None,
        require_report: bool = True,
    ):
        assert mode in ("unimodal", "lupi"), f"Mode tidak valid: {mode}"
        self.data_dir = data_dir
        self.report_dir = report_dir
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        img_col = _detect_column(
            df, ["image_path", "image", "img_path", "filename", "file_name"], "image"
        )
        label_col = _detect_column(
            df, ["label", "target", "class", "tb", "diagnosis"], "label"
        )
        report_col = next(
            (c for c in ["report_path", "report", "txt_path", "text_path"] if c in df.columns),
            None,
        )

        self.samples = []
        skipped_missing_img = 0
        skipped_missing_txt = 0

        for _, row in df.iterrows():
            rel_img = str(row[img_col]).strip()
            img_path = rel_img if os.path.isabs(rel_img) else os.path.join(data_dir, rel_img)
            if not os.path.exists(img_path):
                alt_img = os.path.join(data_dir, "images", os.path.basename(rel_img))
                if os.path.exists(alt_img):
                    img_path = alt_img
                else:
                    skipped_missing_img += 1
                    continue

            label = int(row[label_col])
            report_text = ""

            if self.mode == "lupi" and require_report:
                if report_col is not None and pd.notna(row[report_col]):
                    rel_txt = str(row[report_col]).strip()
                    txt_path = (
                        rel_txt
                        if os.path.isabs(rel_txt)
                        else os.path.join(data_dir, self.report_dir, os.path.basename(rel_txt))
                    )
                    if not os.path.exists(txt_path):
                        txt_path = os.path.join(data_dir, rel_txt)
                else:
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    txt_path = os.path.join(data_dir, self.report_dir, f"{stem}.txt")

                if not os.path.exists(txt_path):
                    skipped_missing_txt += 1
                    continue

                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    report_text = f.read().strip()

                if len(report_text) == 0:
                    skipped_missing_txt += 1
                    continue

            self.samples.append({
                "img_path": img_path,
                "label": label,
                "report_text": report_text,
            })

        if skipped_missing_img > 0 or skipped_missing_txt > 0:
            print(
                f"[Dataset:{mode}] Loaded={len(self.samples)} | "
                f"Skipped missing img={skipped_missing_img}, missing/empty report={skipped_missing_txt}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        image = cv2.imread(item["img_path"])
        if image is None:
            raise RuntimeError(f"Gagal membaca gambar: {item['img_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = torch.tensor(item["label"], dtype=torch.long)

        if self.mode == "lupi" and self.tokenizer is not None and item["report_text"]:
            encoded = self.tokenizer(
                item["report_text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            return {
                "image": image,
                "label": label,
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }

        return {
            "image": image,
            "label": label,
        }