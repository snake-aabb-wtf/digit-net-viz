"""缩放功能验证：滚轮缩放锚点、拖拽平移、双击复位、缩放态下点击命中。"""
from __future__ import annotations

import asyncio

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8123"


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 860})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_function("typeof W !== 'undefined' && W.length === 4", timeout=30000)
        await page.wait_for_timeout(600)

        # 初始视图
        v0 = await page.evaluate("({k: view.k, tx: view.tx, ty: view.ty})")
        print(f"[zoom] 初始视图: {v0}")
        assert v0["k"] == 1 and v0["tx"] == 0 and v0["ty"] == 0

        # 画一个 3（供后续点击测试）
        await page.evaluate("clearPaint()")
        box = await page.locator("#paint").bounding_box()
        pts = [(95, 55), (175, 60), (105, 130), (185, 165), (95, 230)]
        first = True
        for (x, y) in pts:
            px, py = box["x"] + x * box["width"] / 280, box["y"] + y * box["height"] / 280
            if first:
                await page.mouse.move(px, py)
                await page.mouse.down()
                first = False
            else:
                await page.mouse.move(px, py, steps=10)
        await page.mouse.up()
        await page.wait_for_timeout(400)

        # 滚轮缩放（在偏离中心的锚点处），验证锚点不动
        anchor = await page.evaluate(
            """() => { const r = document.getElementById('net').getBoundingClientRect();
                       return { cx: r.left + r.width * 0.55, cy: r.top + r.height * 0.45 }; }"""
        )
        world_before = await page.evaluate(
            f"""() => {{ const r = document.getElementById('net').getBoundingClientRect();
                          const px = ({anchor['cx']} - r.left) * (netW / r.width);
                          const py = ({anchor['cy']} - r.top) * (netH / r.height);
                          const w = screenToWorld(px, py); return {{ x: w.x, y: w.y, px, py }}; }}"""
        )
        await page.mouse.move(anchor["cx"], anchor["cy"])
        await page.mouse.wheel(0, -500)
        await page.wait_for_timeout(200)
        v1 = await page.evaluate("({k: view.k, tx: view.tx, ty: view.ty})")
        world_after = await page.evaluate(
            f"""() => {{ const r = document.getElementById('net').getBoundingClientRect();
                          const px = ({anchor['cx']} - r.left) * (netW / r.width);
                          const py = ({anchor['cy']} - r.top) * (netH / r.height);
                          const w = screenToWorld(px, py); return {{ x: w.x, y: w.y }}; }}"""
        )
        drift = abs(world_after["x"] - world_before["x"]) + abs(world_after["y"] - world_before["y"])
        print(f"[zoom] 缩放后: k={v1['k']:.3f} tx={v1['tx']:.1f} ty={v1['ty']:.1f} · 锚点世界坐标漂移={drift:.4f}px")
        assert v1["k"] > 1.5, "滚轮上滑应放大"
        assert drift < 1.0, "锚点下的世界坐标应保持不动"

        # 缩放态下点击 H2·17 节点：反变换命中
        await page.evaluate(
            """() => { const p = nodePos(2, 17); const c = document.getElementById('net');
                        const r = c.getBoundingClientRect();
                        const sx = (p.x * view.k + view.tx) * r.width / netW;
                        const sy = (p.y * view.k + view.ty) * r.height / netH;
                        c.dispatchEvent(new MouseEvent('click', {clientX: r.left + sx, clientY: r.top + sy, bubbles: true})); }"""
        )
        name = await page.evaluate("document.getElementById('rf-name').textContent")
        print(f"[zoom] 缩放态点击命中: {name}")
        assert "H2 · 神经元 17" in name, "缩放后命中测试应正确反变换"

        # 拖拽平移：先点空白处关闭卡片，再拖拽，确认拖拽的 click 不会误开卡片
        await page.evaluate(
            """() => { const c = document.getElementById('net'); const r = c.getBoundingClientRect();
                        c.dispatchEvent(new MouseEvent('click', {clientX: r.left + r.width/2, clientY: r.top + 6, bubbles: true})); }"""
        )
        await page.wait_for_timeout(100)
        card_before_drag = await page.evaluate("document.getElementById('rf-card').classList.contains('show')")
        print(f"[zoom] 拖拽前卡片已关闭: {not card_before_drag}")
        assert not card_before_drag
        tx0 = (await page.evaluate("({tx: view.tx})"))["tx"]
        await page.mouse.move(anchor["cx"], anchor["cy"])
        await page.mouse.down()
        await page.mouse.move(anchor["cx"] + 70, anchor["cy"] + 40, steps=6)
        await page.mouse.up()
        await page.wait_for_timeout(150)
        v2 = await page.evaluate("({k: view.k, tx: view.tx, ty: view.ty})")
        print(f"[zoom] 拖拽后: tx {tx0:.0f} → {v2['tx']:.0f}")
        assert abs(v2["tx"] - tx0 - 70) < 3, "拖拽应平移视图"
        card_shown_after_drag = await page.evaluate("document.getElementById('rf-card').classList.contains('show')")
        # 拖拽产生的 click 应被吞掉：卡片状态不变
        assert not card_shown_after_drag, "拖拽结束时的 click 不应打开卡片"

        # 双击复位
        await page.mouse.dblclick(anchor["cx"], anchor["cy"])
        await page.wait_for_timeout(150)
        v3 = await page.evaluate("({k: view.k, tx: view.tx, ty: view.ty})")
        print(f"[zoom] 双击复位: {v3}")
        assert v3["k"] == 1 and v3["tx"] == 0 and v3["ty"] == 0

        # 复位后点击仍正常
        await page.evaluate(
            """() => { const p = nodePos(1, 10); const c = document.getElementById('net');
                        const r = c.getBoundingClientRect();
                        const sx = p.x * r.width / netW, sy = p.y * r.height / netH;
                        c.dispatchEvent(new MouseEvent('click', {clientX: r.left + sx, clientY: r.top + sy, bubbles: true})); }"""
        )
        name2 = await page.evaluate("document.getElementById('rf-name').textContent")
        assert "H1 · 神经元 10" in name2, "复位后点击应正常"
        print(f"[zoom] 复位后点击命中: {name2}")

        await page.screenshot(path=r"E:\D\digit-net-viz\shots\zoom.png", full_page=True)
        await browser.close()
        print("[zoom] 页面错误:", errors if errors else "无")
        assert not errors
        print("[zoom] 缩放功能全部通过")


if __name__ == "__main__":
    asyncio.run(main())
