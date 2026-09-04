"""路径高亮验证：选中神经元后，全部五层（含输入像素）都应出现经过它的高亮连线。"""
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
        await page.wait_for_timeout(500)

        # 画一个 3
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

        def click_node(expr: str) -> None:
            return page.evaluate(
                f"""() => {{ const [l, j] = [{expr}]; const p = nodePos(l, j);
                              const c = document.getElementById('net'); const r = c.getBoundingClientRect();
                              const sx = (p.x * view.k + view.tx) * r.width / netW;
                              const sy = (p.y * view.k + view.ty) * r.height / netH;
                              c.dispatchEvent(new MouseEvent('click', {{clientX: r.left + sx, clientY: r.top + sy, bubbles: true}})); }}"""
            )

        # 选中 H2·17：四层连线（784→128→64→32→9 全程）都应有高亮
        await click_node("2, 17")
        await page.wait_for_timeout(100)
        st = await page.evaluate(
            "({sel: [...rfTarget], e: hlEdges.map(s => s.size), n: hlNodes.size, px: hlPixels.size,"
            " card: document.getElementById('rf-card').classList.contains('show')})"
        )
        print(f"[hl] 选中 H2·17: hlEdges各层数量={st['e']} 路径节点={st['n']} 路径像素={st['px']} 卡片={st['card']}")
        assert st["card"] and st["sel"] == [2, 17]
        assert all(v > 0 for v in st["e"]), "H2 神经元的路径应贯穿全部四层连线"
        assert st["px"] > 0, "应回溯到输入像素"

        # 选中 H1·5：下游三层 + 上游输入层；layer1 的高亮线必须以 5 为源
        await click_node("1, 5")
        await page.wait_for_timeout(100)
        st = await page.evaluate(
            "({sel: [...rfTarget], e: hlEdges.map(s => s.size),"
            " srcOk: [...hlEdges[1]].every(k => k.startsWith('5-')),"
            " px: hlPixels.size})"
        )
        print(f"[hl] 选中 H1·5: hlEdges各层数量={st['e']} layer1全部源=5: {st['srcOk']} 路径像素={st['px']}")
        assert st["sel"] == [1, 5]
        assert all(v > 0 for v in st["e"])
        assert st["srcOk"], "H1 神经元的下游高亮线应全部从它出发"

        # 取消选择：点空白 → 高亮清空
        await page.evaluate(
            """() => { const c = document.getElementById('net'); const r = c.getBoundingClientRect();
                        c.dispatchEvent(new MouseEvent('click', {clientX: r.left + r.width/2, clientY: r.top + 6, bubbles: true})); }"""
        )
        await page.wait_for_timeout(100)
        st = await page.evaluate(
            "({sel: rfTarget, e: hlEdges.map(s => s.size), n: hlNodes.size, card: document.getElementById('rf-card').classList.contains('show')})"
        )
        print(f"[hl] 取消选择: rfTarget={st['sel']} 各层高亮={st['e']} 卡片={st['card']}")
        assert st["sel"] is None and all(v == 0 for v in st["e"]) and st["n"] == 0 and not st["card"]

        # 选中禁用神经元：高亮仍应可见（结构展示）
        await page.evaluate(
            """() => { deadMasks[1].add(9); computeHighlight(); }"""
        )
        await click_node("1, 9")
        await page.wait_for_timeout(100)
        st = await page.evaluate(
            "({sel: [...rfTarget], e: hlEdges.map(s => s.size),"
            " srcOk: [...hlEdges[1]].every(k => k.startsWith('9-'))})"
        )
        print(f"[hl] 选中已禁用 H1·9: hlEdges各层数量={st['e']} layer1全部源=9: {st['srcOk']}")
        assert st["sel"] == [1, 9] and all(v > 0 for v in st["e"]) and st["srcOk"]

        await page.screenshot(path=r"E:\D\digit-net-viz\shots\highlight.png", full_page=True)
        await browser.close()
        print("[hl] 页面错误:", errors if errors else "无")
        assert not errors
        print("[hl] 路径高亮全部通过")


if __name__ == "__main__":
    asyncio.run(main())
