"""自动化验证：起服务 → 无头浏览器画 1/3/7 → 核对识别与可视化状态。"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import urllib.request

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8123"
OUT = r"E:\D\digit-net-viz\shots"


def wait_ready() -> None:
    import time
    for _ in range(120):
        try:
            with urllib.request.urlopen(BASE + "/api/status", timeout=2) as r:
                s = json.loads(r.read().decode())
            print(f"[verify] status: source={s['source']} step={s['step']} acc={s['acc']}")
            if s["step"] > 60:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("服务未在 120s 内就绪")


def arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 12):
    return [
        (cx + r * math.sin(math.radians(a)), cy - r * math.cos(math.radians(a)))
        for a in (a0 + (a1 - a0) * i / (n - 1) for i in range(n))
    ]


async def main() -> int:
    wait_ready()
    os.makedirs(OUT, exist_ok=True)

    results: dict[str, dict] = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 860})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_function("document.getElementById('ro-step').textContent !== '—'", timeout=30000)
        await page.wait_for_function("typeof W !== 'undefined' && W.length === 4", timeout=30000)

        strokes: dict[str, list[tuple[int, int]]] = {
            "1": [(130, 45), (140, 105), (128, 215)],
            "3": arc(120, 88, 46, 0, 180) + arc(120, 150, 62, 0, 180),
            "7": [(60, 45), (210, 45), (105, 225)],
        }
        for digit, pts in strokes.items():
            await page.evaluate("clearPaint()")
            box = await page.locator("#paint").bounding_box()
            first = True
            for (x, y) in pts:
                px = box["x"] + x * box["width"] / 280
                py = box["y"] + y * box["height"] / 280
                if first:
                    await page.mouse.move(px, py)
                    await page.mouse.down()
                    first = False
                else:
                    await page.mouse.move(px, py, steps=12)
            await page.mouse.up()
            await page.wait_for_timeout(600)
            top = await page.evaluate(
                "() => { const rows=[...document.querySelectorAll('.bar-row')];"
                "let bi=0; rows.forEach((r,i)=>{if(parseFloat(r.querySelector('.bar-pct').textContent)>parseFloat(rows[bi].querySelector('.bar-pct').textContent))bi=i;});"
                "return {digit: bi+1, pct: rows[bi].querySelector('.bar-pct').textContent}; }"
            )
            results[digit] = top
            print(f"[verify] 画 {digit} → 识别 {top['digit']}（{top['pct']}）")
            await page.screenshot(path=f"{OUT}\\draw_{digit}.png")

        # 感受野交互：用页面自己的 nodePos 算出 H2 层 17 号节点精确坐标后点击
        await page.evaluate(
            "() => { const p = nodePos(2, 17); const c = document.getElementById('net');"
            "const r = c.getBoundingClientRect();"
            "const sx = p.x * r.width / netW, sy = p.y * r.height / netH;"
            "c.dispatchEvent(new MouseEvent('click',{clientX:r.left+sx, clientY:r.top+sy, bubbles:true})); }"
        )
        rf_visible = await page.evaluate("document.getElementById('rf-card').classList.contains('show')")
        rf_name = await page.evaluate("document.getElementById('rf-name').textContent")
        print(f"[verify] 感受野卡片可见: {rf_visible} · {rf_name}")

        await page.screenshot(path=f"{OUT}\\rf.png")

        # 重新训练按钮
        await page.click("#btn-retrain")
        await page.wait_for_timeout(2500)
        epoch_txt = await page.evaluate("document.getElementById('ro-epoch').textContent")
        print(f"[verify] 重新训练后 epoch 读数: {epoch_txt}")

        await page.screenshot(path=f"{OUT}\\final.png", full_page=True)
        await browser.close()

    ok = all(results[d]["digit"] == int(d) for d in ("1", "3", "7"))
    print(f"[verify] 识别结果: {results}")
    print(f"[verify] 页面错误: {errors if errors else '无'}")
    print(f"[verify] 三数字全对: {ok}")
    return 0 if ok and not errors else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

