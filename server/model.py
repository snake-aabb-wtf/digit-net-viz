"""numpy 手写 MLP：784-128-64-32-9，ReLU 隐层 + Softmax 输出，SGD + Momentum。"""
from __future__ import annotations

import numpy as np

ARCH = [784, 128, 64, 32, 9]


class MLP:
    def __init__(self, arch: list[int] | None = None, seed: int = 0) -> None:
        self.arch = arch if arch is not None else ARCH
        rng = np.random.default_rng(seed)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.vW: list[np.ndarray] = []
        self.vb: list[np.ndarray] = []
        for fin, fout in zip(self.arch[:-1], self.arch[1:]):
            w = (rng.standard_normal((fin, fout)) * np.sqrt(2.0 / fin)).astype(np.float32)
            self.W.append(w)
            self.b.append(np.zeros(fout, np.float32))
            self.vW.append(np.zeros_like(w))
            self.vb.append(np.zeros(fout, np.float32))

    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        """返回各层激活 [x, h1, h2, h3, prob]。"""
        acts = [x]
        a = x
        last = len(self.W) - 1
        for i in range(last + 1):
            z = a @ self.W[i] + self.b[i]
            if i == last:
                z = z - z.max(axis=1, keepdims=True)
                e = np.exp(z)
                a = e / e.sum(axis=1, keepdims=True)
            else:
                a = np.maximum(z, 0.0)
            acts.append(a)
        return acts

    def train_step(self, xb: np.ndarray, yb: np.ndarray, lr: float, mu: float = 0.9) -> float:
        """一个小批量，返回交叉熵损失。yb 为 0..8 整数标签。"""
        acts = self.forward(xb)
        out = acts[-1]
        n = xb.shape[0]
        rows = np.arange(n)
        loss = float(-np.log(out[rows, yb] + 1e-9).mean())

        delta = out.copy()
        delta[rows, yb] -= 1.0
        delta /= np.float32(n)

        for i in range(len(self.W) - 1, -1, -1):
            gW = acts[i].T @ delta
            gb = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
            self.vW[i] = mu * self.vW[i] - lr * gW
            self.vb[i] = mu * self.vb[i] - lr * gb
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]
        return loss

    def accuracy(self, x: np.ndarray, y: np.ndarray, batch: int = 2000) -> float:
        correct = 0
        for s in range(0, x.shape[0], batch):
            out = self.forward(x[s : s + batch])[-1]
            correct += int((out.argmax(axis=1) == y[s : s + batch]).sum())
        return correct / max(1, x.shape[0])
