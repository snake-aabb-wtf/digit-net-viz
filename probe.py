"""像素级体检：无头浏览器里采样各画布区域，确认渲染内容与布局合理。"""
from __future__ import annotations

import asyncio
import json
import urllib.request

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8123"


async def main() -> None:
    with urllib.request.urlopen(BASE + "/api/status", timeout=2) as r:
        print("[probe] status:", r.read().decode())

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 860})
        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_function("typeof W !== 'undefined' && W.length === 4", timeout=30000)
        await page.wait_for_timeout(1500)

        info = await page.evaluate(
            """() => {
              const cv = document.getElementById('net');
              const c = colX();
              const ctx = cv.getContext('2d');
              const dpr = devicePixelRatio || 1;
              const img = ctx.getImageData(0, 0, cv.width, cv.height).data;
              const colHit = (x0, x1) => {
                let n = 0;
                for (let y = 0; y < cv.height; y += 4)
                  for (let x = Math.floor(x0*dpr); x < Math.floor(x1*dpr); x += 4) {
                    const a = img[(y * cv.width + x) * 4 + 3];
                    if (a > 10) n++;
                  }
                return n;
              };
              return {
                netW, netH,
                gx: c.gx, gridSide: c.gridSide,
                hiddenX: c.hiddenX, outX: c.outX,
                hits: {
                  grid: colHit(c.gx, c.gx + c.gridSide),
                  h1: colHit(c.hiddenX[0] - 12, c.hiddenX[0] + 12),
                  h2: colHit(c.hiddenX[1] - 12, c.hiddenX[1] + 12),
                  h3: colHit(c.hiddenX[2] - 12, c.hiddenX[2] + 12),
                  out: colHit(c.outX - 14, c.outX + 14),
                },
                bars: [...document.querySelectorAll('.bar-fill')].map(b => b.style.width),
                chartPx: (() => {
                  const ch = document.getElementById('chart');
                  const d = ch.getContext('2d').getImageData(0, 0, ch.width, ch.height).data;
                  let n = 0;
                  for (let i = 3; i < d.length; i += 4) if (d[i] > 10) n++;
                  return n;
                })(),
              };
            }"""
        )
        print("[probe] layout:", json.dumps(info["hits"]), "netW/H:", info["netW"], info["netH"])
        print("[probe] cols: gx", round(info["gx"]), "gridSide", round(info["gridSide"]),
              "hiddenX", [round(v) for v in info["hiddenX"]], "outX", round(info["outX"]))
        for k, v in info["hits"].items():
            assert v > 50, f"画布区域 {k} 疑似空白（{v} 像素）"
        assert abs(info["hiddenX"][0] - info["hiddenX"][1]) > 40, "隐层列仍然挤在一起"
        assert info["chartPx"] > 200, "训练曲线为空"
        print("[probe] 各列渲染 OK · 训练曲线 OK · 布局分布 OK")

        # 画 5 并检查热图/条形图联动
        await page.evaluate("clearPaint()")
        box = await page.locator("#paint").bounding_box()
        pts = [(150, 45), (150, 45), (110, 90), (190, 90), (110, 160), (190, 160), (150, 215), (150, 215)]
        first = True
        for (x, y) in pts:
            px = box["x"] + x * box["width"] / 280
            py = box["y"] + y * box["height"] / 280
            if first:
                await page.mouse.move(px, py)
                await page.mouse.down()
                first = False
            else:
                await page.mouse.move(px, py, steps=10)
        await page.mouse.up()
        await page.wait_for_timeout(500)
        heat = await page.evaluate(
            """() => {
              const c = colX();
              const cv = document.getElementById('net');
              const dpr = devicePixelRatio || 1;
              const img = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
              let lit = 0;
              for (let i = 0; i < 784; i++) {
                const x = c.gx + (i % 28) * c.gridSide / 28, y = c.gy + Math.floor(i / 28) * c.gridSide / 28;
                const px = Math.floor(x * dpr), py = Math.floor(y * dpr);
                const a = img[(py * cv.width + px) * 4 + 3];
                if (a > 40) lit++;
              }
              return lit;
            }"""
        )
        bars = await page.evaluate("[...document.querySelectorAll('.bar-fill')].map(b => b.style.width)")
        verdict = await page.evaluate("document.getElementById('verdict').textContent")
        print(f"[probe] 热图点亮像素采样: {heat}/784 · 条形图: {bars} · 判定: {verdict}")
        assert heat > 25, "输入热图没有点亮"
        assert any(w not in ("0%", "") for w in bars), "条形图没有更新"
        print("[probe] 全部体检通过")

        await page.screenshot(path=r"E:\D\digit-net-viz\shots\probe.png", full_page=True)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
