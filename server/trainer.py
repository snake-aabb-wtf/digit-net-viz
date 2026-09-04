"""后台训练线程：周期性生成「权重快照 + 指标」，供 WebSocket 推送。"""
from __future__ import annotations

import base64
import threading
import time

import numpy as np

from .data import load_dataset
from .model import MLP

BATCH = 64
EPOCHS = 18
LR0 = 0.10
LR_DECAY = 0.90
TICK_SECONDS = 0.7


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode("ascii")


class Trainer:
    """在守护线程里训练；发布带自增 seq 的最新快照（单槽，无积压）。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._latest: dict | None = None
        self._seq = 0
        self.stop_event = threading.Event()
        self.retrain_event = threading.Event()

        print("[trainer] 正在准备数据（首次运行会下载 MNIST，约 12 MB）...", flush=True)
        x, y, xt, yt, self.source = load_dataset()
        self.x, self.y, self.xt, self.yt = x, y, xt, yt
        print(f"[trainer] 数据就绪：source={self.source}  train={x.shape[0]}  test={xt.shape[0]}", flush=True)

        self.thread = threading.Thread(target=self._run, daemon=True, name="trainer")
        self.thread.start()

    def snapshot(self) -> dict | None:
        with self.lock:
            return self._latest

    def request_retrain(self) -> None:
        self.retrain_event.set()

    def _publish(self, model: MLP, step: int, epoch: int, loss, acc, done: bool) -> None:
        with self.lock:
            self._latest = {
                "seq": self._seq,
                "step": int(step),
                "epoch": int(epoch),
                "loss": None if loss is None else round(float(loss), 4),
                "acc": None if acc is None else round(float(acc), 4),
                "done": bool(done),
                "W": [b64(w) for w in model.W],
                "b": [b64(b) for b in model.b],
            }
            self._seq += 1

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.retrain_event.clear()
            model = MLP(seed=int(np.random.randint(0, 2**31)))
            self._publish(model, 0, 0, None, model.accuracy(self.xt, self.yt), False)
            print("[trainer] 权重已重新初始化，开始训练", flush=True)

            rng = np.random.default_rng(1234)
            n = self.x.shape[0]
            step = 0
            loss_ema = None
            t_last = time.time()
            aborted = False

            for epoch in range(1, EPOCHS + 1):
                lr = LR0 * (LR_DECAY ** (epoch - 1))
                order = rng.permutation(n)
                for s in range(0, n, BATCH):
                    if self.retrain_event.is_set() or self.stop_event.is_set():
                        aborted = True
                        break
                    idx = order[s : s + BATCH]
                    loss = model.train_step(self.x[idx], self.y[idx], lr)
                    step += 1
                    loss_ema = loss if loss_ema is None else 0.95 * loss_ema + 0.05 * loss
                    now = time.time()
                    if now - t_last >= TICK_SECONDS:
                        t_last = now
                        acc = model.accuracy(self.xt, self.yt)
                        self._publish(model, step, epoch, loss_ema, acc, False)
                if aborted:
                    break

            if not aborted and not self.stop_event.is_set():
                acc = model.accuracy(self.xt, self.yt)
                self._publish(model, step, EPOCHS, loss_ema, acc, True)
                print(f"[trainer] 训练完成：step={step}  test_acc={acc:.4f}", flush=True)
                self.retrain_event.wait()  # 闲置，等待「重新训练」请求
