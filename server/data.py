"""数据加载：优先在线拉取 MNIST（自动缓存），失败则用字体合成数据兜底。

标签统一重映射到 0..8（对应数字 1..9，按需求不包含 0）。
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

TRAIN_KEEP = 24000
TEST_KEEP = 2000


def _download(name: str) -> bytes:
    last_err: Exception | None = None
    for base in MIRRORS:
        url = base + name
        try:
            print(f"[data] 下载 {url} ...", flush=True)
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[data] 镜像失败：{e}", flush=True)
    raise RuntimeError(f"无法下载 {name}: {last_err}")


def _parse_images(blob: bytes) -> np.ndarray:
    data = gzip.decompress(blob)
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"images magic 不匹配: {magic}")
    arr = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows * cols)
    return arr.astype(np.float32) / 255.0


def _parse_labels(blob: bytes) -> np.ndarray:
    data = gzip.decompress(blob)
    magic, n = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"labels magic 不匹配: {magic}")
    return np.frombuffer(data, dtype=np.uint8, offset=8).copy()


def _cache_path(name: str) -> str:
    return os.path.join(DATA_DIR, name + ".npy")


def _load_cached(name: str) -> np.ndarray | None:
    p = _cache_path(name)
    return np.load(p) if os.path.exists(p) else None


def _save_cache(name: str, arr: np.ndarray) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(_cache_path(name), arr)


def _center_like_mnist(img, rng: np.random.Generator | None = None) -> np.ndarray:
    """把 PIL 图像裁剪、等比缩到 20px、按质心居中放进 28×28（与 MNIST 规格对齐）。"""
    a = np.asarray(img, dtype=np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.15)
    if ys.size == 0:
        return np.zeros((28, 28), np.float32)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = img.crop((x0, y0, x1, y1))
    w, h = crop.size
    s = 20.0 / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    crop = crop.resize((nw, nh), Image.BILINEAR)
    ca = np.asarray(crop, dtype=np.float32) / 255.0
    yy, xx = np.nonzero(ca > 0.15)
    cy, cx = float(yy.mean()), float(xx.mean())
    dx, dy = int(round(14 - cx)), int(round(14 - cy))
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop, (dx, dy))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _synthetic_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """兜底方案：多字体渲染数字 1..9 + 旋转/平移/缩放/加粗/模糊增广。"""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    windir = os.environ.get("WINDIR", r"C:\Windows")
    font_files = []
    for cand in (
        "arial.ttf", "calibri.ttf", "segoeui.ttf", "tahoma.ttf",
        "verdana.ttf", "times.ttf", "georgia.ttf", "consola.ttf", "couri.ttf",
    ):
        p = os.path.join(windir, "Fonts", cand)
        if os.path.exists(p):
            font_files.append(p)
    if not font_files:
        raise RuntimeError("找不到任何系统字体，合成数据不可用")

    def make(n: int, seed: int):
        rng = np.random.default_rng(seed)
        imgs = np.zeros((n, 784), np.float32)
        labels = rng.integers(0, 9, n).astype(np.int64)  # 0..8 → 数字 1..9
        for i in range(n):
            digit = int(labels[i]) + 1
            size = int(rng.integers(160, 230))
            font = ImageFont.truetype(font_files[int(rng.integers(0, len(font_files)))], size)
            img = Image.new("L", (280, 280), 0)
            ImageDraw.Draw(img).text((140, 140), str(digit), font=font, fill=255, anchor="mm")
            img = img.rotate(
                rng.uniform(-14, 14), resample=Image.BILINEAR,
                translate=(rng.uniform(-10, 10), rng.uniform(-10, 10)),
                scale=float(rng.uniform(0.82, 1.05)),
            )
            if rng.random() < 0.5:
                img = img.filter(ImageFilter.MaxFilter(3))
            if rng.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.2)))
            imgs[i] = _center_like_mnist(img).reshape(784)
        return imgs, labels

    xtr, ytr = make(6000, 1)
    xte, yte = make(800, 2)
    return xtr, ytr, xte, yte


def load_dataset():
    """返回 (x_train, y_train, x_test, y_test, source)。像素 0..1，标签 0..8。"""
    parts = {k: _load_cached(k) for k in FILES}
    if any(v is None for v in parts.values()):
        try:
            for k, fname in FILES.items():
                if parts[k] is None:
                    raw = _download(fname)
                    arr = _parse_images(raw) if "images" in k else _parse_labels(raw)
                    _save_cache(k, arr)
                    parts[k] = arr
            source = "mnist"
        except Exception as e:  # noqa: BLE001
            print(f"[data] MNIST 获取失败（{e}），使用字体合成数据兜底。", flush=True)
            xtr, ytr, xte, yte = _synthetic_dataset()
            return xtr, ytr, xte, yte, "synthetic"

    x_tr, y_tr = parts["train_images"], parts["train_labels"]
    x_te, y_te = parts["test_images"], parts["test_labels"]
    tr_mask = (y_tr >= 1) & (y_tr <= 9)
    te_mask = (y_te >= 1) & (y_te <= 9)
    x_tr, y_tr = x_tr[tr_mask], y_tr[tr_mask].astype(np.int64) - 1
    x_te, y_te = x_te[te_mask], y_te[te_mask].astype(np.int64) - 1

    rng = np.random.default_rng(7)
    tr_idx = rng.permutation(x_tr.shape[0])[:TRAIN_KEEP]
    te_idx = rng.permutation(x_te.shape[0])[:TEST_KEEP]
    return (
        np.ascontiguousarray(x_tr[tr_idx]),
        np.ascontiguousarray(y_tr[tr_idx]),
        np.ascontiguousarray(x_te[te_idx]),
        np.ascontiguousarray(y_te[te_idx]),
        source,
    )
