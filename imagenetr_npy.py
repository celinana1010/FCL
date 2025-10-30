import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------- scanning / splits ----------

def scan_imagenet_r(root: Path) -> List[Tuple[str, str]]:
    items = []
    for cls in sorted(p.name for p in root.iterdir() if p.is_dir()):
        cdir = root / cls
        for fp in cdir.rglob("*"):
            if fp.suffix.lower() in IMG_EXTS:
                items.append((str(fp), cls))
    if not items:
        raise RuntimeError(f"No images found under {root}.")
    return items


def build_label_map(items: List[Tuple[str, str]]):
    classes = sorted({c for _, c in items})
    cls2id = {c: i for i, c in enumerate(classes)}
    return cls2id


def stratified_split(
    pairs: List[Tuple[str, int]],
    test_ratio: float = 0.1,
    seed: int = 0,
):
    random.seed(seed)
    by_cls = defaultdict(list)
    for p, y in pairs:
        by_cls[y].append(p)
    train, test = [], []
    for y, arr in by_cls.items():
        random.shuffle(arr)
        n = len(arr)
        nt = max(1, int(round(n * test_ratio)))
        test += [(p, y) for p in arr[:nt]]
        train += [(p, y) for p in arr[nt:]]
    return train, test


# ---------- dataset / transforms ----------

def build_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageList(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, y


# ---------- backbones (feature extractors) ----------

class GlobalPool(nn.Module):
    """Global average pool: (N, C, H, W) -> (N, C)"""
    def __init__(self): super().__init__()
    def forward(self, x):  # x: N,C,H,W
        return torch.flatten(torch.mean(x, dim=[2, 3], keepdim=False), 1)

def build_backbone(name: str):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = m.fc.in_features
        m = nn.Sequential(*(list(m.children())[:-2]), GlobalPool())
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = m.fc.in_features
        m = nn.Sequential(*(list(m.children())[:-2]), GlobalPool())
    elif name in {"vit_b_16", "vit-b-16"}:
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # vit returns (N, D) from forward_features if we bypass the head
        class ViTFeatures(nn.Module):
            def __init__(self, vit): super().__init__(); self.vit = vit
            def forward(self, x):
                # forward_features applies patch+pos+encoder then returns CLS token embedding
                return self.vit.forward_features(x)
        m = ViTFeatures(m)
        feat_dim = 768
    else:
        raise ValueError(f"Unknown model: {name}")
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, feat_dim


# ---------- encoding & dumping ----------

@torch.no_grad()
def encode(items, model, img_size: int, batch: int, num_workers: int, device: str):
    ds = ImageList(items, build_eval_transform(img_size))
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    feats, labels = [], []
    for imgs, ys in dl:
        imgs = imgs.to(device, non_blocking=True)
        z = model(imgs)
        if z.ndim > 2:  # safety: if someone returns N,C,H,W
            z = torch.flatten(torch.mean(z, dim=[2, 3], keepdim=False), 1)
        feats.append(z.cpu().float())
        labels.append(ys.long())
    feats = torch.cat(feats, dim=0).numpy().astype("float32")
    labels = torch.cat(labels, dim=0).numpy().astype("int64")
    return feats, labels


def dump_per_class(feats: np.ndarray, labels: np.ndarray, out_dir: Path, K: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = [[] for _ in range(K)]
    for f, y in zip(feats, labels):
        buckets[y].append(f)
    counts = []
    for k in range(K):
        arr = np.stack(buckets[k], axis=0) if buckets[k] else np.zeros((0, feats.shape[1]), dtype="float32")
        np.save(out_dir / f"{k}.npy", arr)
        counts.append(len(arr))
    return counts


# ---------- main ----------

def main(args):
    root = Path(args.root)
    items = scan_imagenet_r(root)
    cls2id = build_label_map(items)
    pairs = [(p, cls2id[c]) for p, c in items]

    train, test = stratified_split(pairs, test_ratio=args.test_ratio, seed=args.seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, feat_dim = build_backbone(args.model)
    model.to(device)

    print(f"Found {len(cls2id)} classes, {len(pairs)} images. Using {args.model} → {feat_dim}-D features on {device}.")
    print("Encoding TRAIN...")
    tr_feats, tr_labels = encode(train, model, args.img_size, args.batch, args.num_workers, device)
    print("Encoding TEST...")
    te_feats, te_labels = encode(test,  model, args.img_size, args.batch, args.num_workers, device)

    out_train = Path(args.out_train)
    out_test  = Path(args.out_test)
    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    print("Saving per-class .npy (train)...")
    tr_counts = dump_per_class(tr_feats, tr_labels, out_train, K=len(cls2id))
    print("Saving per-class .npy (test)...")
    te_counts = dump_per_class(te_feats, te_labels, out_test,  K=len(cls2id))

    meta = {
        "cls2id": cls2id, "num_classes": len(cls2id),
        "feature_dim": feat_dim, "model": args.model, "img_size": args.img_size,
        "split": {"train_total": int(tr_feats.shape[0]), "test_total": int(te_feats.shape[0]),
                  "train_counts": tr_counts, "test_counts": te_counts},
        "seed": args.seed,
    }
    with open(out_train / "meta.json", "w") as f: json.dump(meta, f, indent=2)
    print(f"Done. Wrote meta.json to {out_train}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to ImageNet-R root (folders of classes).")
    ap.add_argument("--out-train", default="dataset/imagenetr-classes")
    ap.add_argument("--out-test",  default="dataset/imagenetr-test-classes")
    ap.add_argument("--model", default="resnet50",
                    choices=["resnet18", "resnet50", "vit_b_16"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
