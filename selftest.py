"""静态模式自检：在无 WS 的纯静态目录上验证自动降级、回放训练与识别。

用法：python selftest.py --dist dist
（本地 web/ 缺文件时会拷入 dist；CI 中 dist 已就绪时跳过）
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import os
import shutil
import sys
import threading
import time

from playwright.async_api import async_playwright

PORT = 8900
BASE = f"http://127.0.0.1:{PORT}"


def start_server(dist: str):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=dist)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="dist")
    args = ap.parse_args()

    for f in ("index.html", "style.css", "main.js"):
        src = os.path.join("web", f)
        dst = os.path.join(args.dist, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    for f in ("weights.json", "timeline.json"):
        assert os.path.exists(os.path.join(args.dist, f)), f"缺少 {f}（先跑 python -m server.train_export）"

    httpd = start_server(args.dist)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 860})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text)
                if m.type == "error" and "WebSocket" not in m.text else None)  # 静态站上 WS 404 是预期的降级信号

        await page.goto(BASE, wait_until="networkidle")
        lamp_el = page.wait_for_function(
            "document.getElementById('lamp-text').textContent.includes('静态')", timeout=30000
        )
        print("[selftest] 进入静态模式:", await (await lamp_el).evaluate("document.getElementById('lamp-text').textContent"))
        btn = await page.evaluate("document.getElementById('btn-retrain').textContent")
        assert btn == "回放训练", f"按钮应为「回放训练」，实际「{btn}」"

        # 回放训练：等待按钮恢复可用且文案复原
        t0 = time.time()
        await page.click("#btn-retrain")
        await page.wait_for_function(
            "document.getElementById('btn-retrain').textContent === '回放训练'"
            " && !document.getElementById('btn-retrain').disabled", timeout=180000
        )
        lamp2 = await page.evaluate("document.getElementById('lamp-text').textContent")
        acc = await page.evaluate("document.getElementById('ro-acc').textContent")
        print(f"[selftest] 回放完成（{time.time()-t0:.0f}s）: {lamp2} · 最终 acc={acc}")

        # 回放后画 7 识别
        await page.evaluate("clearPaint()")
        box = await page.locator("#paint").bounding_box()
        pts = [(60, 45), (210, 45), (105, 225)]
        first = True
        for (x, y) in pts:
            px, py = box["x"] + x * box["width"] / 280, box["y"] + y * box["height"] / 280
            if first:
                await page.mouse.move(px, py)
                await page.mouse.down()
                first = False
            else:
                await page.mouse.move(px, py, steps=12)
        await page.mouse.up()
        await page.wait_for_timeout(600)
        top = await page.evaluate("probs.indexOf(Math.max(...probs)) + 1")
        print(f"[selftest] 画 7 → 识别 {top}")
        assert top == 7

        os.makedirs("shots", exist_ok=True)
        await page.screenshot(path="shots/selftest.png", full_page=True)
        await browser.close()
        print("[selftest] 页面错误:", errors if errors else "无")
        assert not errors

    httpd.shutdown()
    print("[selftest] 静态模式全部通过")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(asyncio.run(main()))
