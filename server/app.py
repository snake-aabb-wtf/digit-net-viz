"""FastAPI 入口：/ws 推送训练快照，/api/status 状态查询，静态托管 web/。"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .model import ARCH
from .trainer import EPOCHS, Trainer

WEB_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "web"))
HOST = "127.0.0.1"
PORT = 8123

trainer: Trainer | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global trainer
    trainer = Trainer()
    yield


app = FastAPI(lifespan=lifespan, title="digit-net-viz")


@app.get("/api/status")
async def status():
    snap = trainer.snapshot() if trainer else None
    return {
        "source": trainer.source if trainer else None,
        "arch": ARCH,
        "epochs": EPOCHS,
        "step": snap["step"] if snap else 0,
        "epoch": snap["epoch"] if snap else 0,
        "loss": snap["loss"] if snap else None,
        "acc": snap["acc"] if snap else None,
        "done": snap["done"] if snap else False,
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    assert trainer is not None
    snap = trainer.snapshot()
    await ws.send_text(json.dumps({
        "type": "init",
        "arch": ARCH,
        "source": trainer.source,
        "trainN": int(trainer.x.shape[0]),
        "testN": int(trainer.xt.shape[0]),
        "epochs": EPOCHS,
        "snapshot": snap,
    }))
    last_seq = snap["seq"] if snap else -1

    async def reader() -> None:
        while True:
            msg = await ws.receive_text()
            if msg.strip() == "retrain":
                print("[ws] 收到重新训练请求", flush=True)
                trainer.request_retrain()

    async def writer() -> None:
        nonlocal last_seq
        while True:
            await asyncio.sleep(0.7)
            snap = trainer.snapshot()
            if snap and snap["seq"] != last_seq:
                last_seq = snap["seq"]
                await ws.send_text(json.dumps({"type": "tick", **snap}))

    rd = asyncio.create_task(reader())
    wr = asyncio.create_task(writer())
    try:
        await rd
    except WebSocketDisconnect:
        pass
    finally:
        wr.cancel()
        try:
            await wr
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    print(f" 手写数字 · 神经网络观测台  →  http://{HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
