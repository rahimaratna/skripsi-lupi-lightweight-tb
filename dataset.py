import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size: int = 224, is_train: bool = True):
    if is_train:
        # Sesuai hlm. 3 paper: CLAHE baku diterapkan pada pipeline training
        return A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def read_split_csv(csv_path: str) -> pd.DataFrame:
    """
    Membaca file CSV split dan mendeteksi otomatis jika file tidak memiliki baris header.
    """
    df = pd.read_csv(csv_path)
    if len(df.columns) >= 2:
        first_col = str(df.columns[0]).strip().lower()
        second_col = str(df.columns[1]).strip()
        known_headers = {"image_path", "image", "img_path", "filename", "file_name"}

        if first_col not in known_headers and (
            first_col.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            or second_col in ("0", "1")
        ):
            df = pd.read_csv(csv_path, header=None)
            if df.shape[1] == 2:
                df.columns = ["image_path", "label"]
            elif df.shape[1] >= 3:
                df.columns = ["image_path", "label", "report_path"] + [
                    f"col_{i}" for i in range(3, df.shape[1])
                ]
    return df


def _detect_column(df: pd.DataFrame, candidates: list[str], col_desc: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Kolom {col_desc} tidak ditemukan. Kolom tersedia: {list(df.columns)}")


def _build_file_index(root_dir: str, valid_exts: tuple[str, ...]) -> dict[str, str]:
    index = {}
    if not os.path.exists(root_dir):
        return index
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(valid_exts):
                index[fname] = os.path.join(dirpath, fname)
    return index


class TBMultimodalDataset(Dataset):
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

        img_index = _build_file_index(
            os.path.join(data_dir, "images"), (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        )
        txt_index = (
            _build_file_index(os.path.join(data_dir, report_dir), (".txt",))
            if (self.mode == "lupi" and require_report)
            else {}
        )

        self.samples = []
        skipped_missing_img = 0
        skipped_missing_txt = 0

        for _, row in df.iterrows():
            rel_img = str(row[img_col]).strip()
            base_img = os.path.basename(rel_img)

            if os.path.isabs(rel_img) and os.path.exists(rel_img):
                img_path = rel_img
            elif os.path.exists(os.path.join(data_dir, rel_img)):
                img_path = os.path.join(data_dir, rel_img)
            elif base_img in img_index:
                img_path = img_index[base_img]
            else:
                skipped_missing_img += 1
                continue

            label = int(row[label_col])
            report_text = ""

            if self.mode == "lupi" and require_report:
                if report_col is not None and pd.notna(row[report_col]):
                    rel_txt = str(row[report_col]).strip()
                    base_txt = os.path.basename(rel_txt)
                else:
                    stem = os.path.splitext(base_img)[0]
                    rel_txt = os.path.join(report_dir, f"{stem}.txt")
                    base_txt = f"{stem}.txt"

                if os.path.isabs(rel_txt) and os.path.exists(rel_txt):
                    txt_path = rel_txt
                elif os.path.exists(os.path.join(data_dir, report_dir, base_txt)):
                    txt_path = os.path.join(data_dir, report_dir, base_txt)
                elif os.path.exists(os.path.join(data_dir, rel_txt)):
                    txt_path = os.path.join(data_dir, rel_txt)
                elif base_txt in txt_index:
                    txt_path = txt_index[base_txt]
                else:
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