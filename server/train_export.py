"""无服务的训练导出：跑完整训练，输出静态部署所需的 weights.json 与 timeline.json。

用法：
    python -m server.train_export --out dist

产物：
    dist/weights.json   最终权重（f32）+ 元信息，前端「静态模式」直接加载
    dist/timeline.json  训练过程时间线（int8 量化权重快照），前端「回放训练」用
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os

import numpy as np

from .data import load_dataset
from .model import MLP

BATCH = 64
EPOCHS = 18
LR0 = 0.10
LR_DECAY = 0.90
FRAME_EVERY = 100  # 每 100 步存一帧时间线（约 68 帧 ≈ 10MB）


def b64_f32(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode("ascii")


def b64_i8(a: np.ndarray) -> tuple[float, str]:
    """按张量最大绝对值量化到 int8，返回 (scale, b64)；scale = max|a|/127，还原时 q*scale≈a。"""
    mx = float(np.max(np.abs(a))) or 1.0
    s = mx / 127.0
    q = np.clip(np.round(a / s), -127, 127).astype(np.int8)
    return s, base64.b64encode(q.tobytes()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist", help="输出目录")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("[export] 正在准备数据 ...", flush=True)
    x, y, xt, yt, source = load_dataset()
    model = MLP(seed=7)  # 固定种子，CI 产物可复现
    rng = np.random.default_rng(1234)
    n = x.shape[0]

    frames = []
    step = 0
    loss_ema = None
    for epoch in range(1, EPOCHS + 1):
        lr = LR0 * (LR_DECAY ** (epoch - 1))
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s : s + BATCH]
            loss = model.train_step(x[idx], y[idx], lr)
            step += 1
            loss_ema = loss if loss_ema is None else 0.95 * loss_ema + 0.05 * loss
            if step % FRAME_EVERY == 0:
                acc = model.accuracy(xt, yt)
                frames.append({
                    "step": step, "epoch": epoch,
                    "loss": round(float(loss_ema), 4), "acc": round(float(acc), 4),
                    "Wq": [dict(zip(("s", "d"), b64_i8(w))) for w in model.W],
                    "b": [b64_f32(b) for b in model.b],
                })
                print(f"[export] frame {len(frames):3d}  step={step:5d}  epoch={epoch:2d}  "
                      f"loss={loss_ema:.4f}  acc={acc:.4f}", flush=True)

    final_acc = model.accuracy(xt, yt)
    weights = {
        "arch": model.arch,
        "source": source,
        "trainedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "repo": os.environ.get("GITHUB_REPOSITORY", ""),
        "epochs": EPOCHS,
        "steps": step,
        "trainN": int(x.shape[0]),
        "testN": int(xt.shape[0]),
        "loss0": frames[0]["loss"],
        "acc0": frames[0]["acc"],
        "loss": round(float(loss_ema), 4),
        "acc": round(float(final_acc), 4),
        "W": [b64_f32(w) for w in model.W],
        "b": [b64_f32(b) for b in model.b],
    }
    timeline = {"arch": model.arch, "frameMs": 350, "frames": frames}

    wp = os.path.join(args.out, "weights.json")
    tp = os.path.join(args.out, "timeline.json")
    with open(wp, "w", encoding="utf-8") as f:
        json.dump(weights, f)
    with open(tp, "w", encoding="utf-8") as f:
        json.dump(timeline, f)
    print(f"[export] 完成：{wp} ({os.path.getsize(wp)/1e6:.2f} MB)  "
          f"{tp} ({os.path.getsize(tp)/1e6:.2f} MB)  test_acc={final_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
