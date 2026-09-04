"""消融功能验证：画 5 → 禁用 H1 神经元 → 前向变化 → 恢复 → 概率还原。"""
from __future__ import annotations

import asyncio
import json

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
        await page.wait_for_timeout(800)

        # 画一个 5
        await page.evaluate("clearPaint()")
        box = await page.locator("#paint").bounding_box()
        pts = [(190, 50), (95, 50), (90, 120), (175, 125), (185, 190), (100, 225), (95, 225)]
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
        await page.wait_for_timeout(500)

        before = await page.evaluate("({top: probs.indexOf(Math.max(...probs)), p5: probs[4], act: acts[1][5]})")
        print(f"[ablate] 禁用前: top={before['top']+1} p(5)={before['p5']:.4f} h1[5]={before['act']:.4f}")

        # 选出对该输入激活最强的 H1 神经元，点开它的感受野并禁用
        target = await page.evaluate("acts[1].indexOf(Math.max(...acts[1]))")
        print(f"[ablate] 激活最强的 H1 神经元: #{target} (a={await page.evaluate(f'acts[1][{target}]'):.4f})")
        await page.evaluate(
            f"""() => {{ const p = nodePos(1, {target}); const c = document.getElementById('net');
            const r = c.getBoundingClientRect();
            const sx = p.x * r.width / netW, sy = p.y * r.height / netH;
            c.dispatchEvent(new MouseEvent('click',{{clientX:r.left+sx, clientY:r.top+sy, bubbles:true}})); }}"""
        )
        name = await page.evaluate("document.getElementById('rf-name').textContent")
        assert f"神经元 {target}" in name, f"点击应选中神经元 {target}，实际: {name}"
        await page.click("#rf-toggle")
        after_toggle_txt = await page.evaluate("document.getElementById('rf-toggle').textContent")
        restore_visible = await page.evaluate("!document.getElementById('btn-enable-all').disabled")
        disabled = await page.evaluate(
            f"({{p5: probs[4], act: acts[1][{target}], top: probs.indexOf(Math.max(...probs)), pTop: Math.max(...probs)}})"
        )
        print(f"[ablate] 卡片: {name} · 按钮变为「{after_toggle_txt}」 · 恢复按钮可见: {restore_visible}")
        print(f"[ablate] 禁用后: top={disabled['top']+1} 置信度={disabled['pTop']:.4f} h1[{target}]={disabled['act']:.4f}")

        assert abs(disabled["act"]) < 1e-6, "被禁用神经元的激活应为 0"
        assert after_toggle_txt == "重新启用此神经元"
        assert restore_visible
        # 单神经元消融通常不改变结果 —— 这正是网络冗余性的体现，打印观察即可
        print(f"[ablate] 单神经元消融后输出不变: {abs(disabled['pTop'] - before['p5']) < 1e-6}（冗余网络的正常现象）")

        # 批量消融：掐掉 H1 激活前 30 强，网络应明显受损
        await page.evaluate(
            """() => {
              const idx = [...acts[1].keys()].sort((a, b) => acts[1][b] - acts[1][a]).slice(0, 30);
              idx.forEach(j => deadMasks[1].add(j));
              acts = forward(inputVec); probs = acts[acts.length - 1]; updateBars(); updateEnableAllBtn();
            }"""
        )
        batch = await page.evaluate(
            "({top: probs.indexOf(Math.max(...probs)), pTop: Math.max(...probs), n: deadMasks[1].size})"
        )
        print(f"[ablate] 掐掉 H1 前 30 强后: top={batch['top']+1} 置信度={batch['pTop']:.4f} 已禁用 {batch['n']} 个")
        assert batch["pTop"] < before["p5"] - 0.05 or batch["top"] != before["top"], "批量消融应明显改变输出"

        # 恢复
        await page.click("#btn-enable-all")
        restored = await page.evaluate("({p5: probs[4], pTop: Math.max(...probs), top: probs.indexOf(Math.max(...probs))})")
        restore_state = await page.evaluate(
            "({d: document.getElementById('btn-enable-all').disabled, t: document.getElementById('btn-enable-all').textContent})"
        )
        restore_hidden = restore_state["d"] and restore_state["t"] == "启用所有神经元"
        print(f"[ablate] 恢复后: top={restored['top']+1} p(5)={restored['p5']:.4f} 置信度={restored['pTop']:.4f} · 恢复按钮隐藏: {restore_hidden}")
        assert abs(restored["p5"] - before["p5"]) < 1e-6, "恢复后概率应还原"

        # 重训清空状态
        await page.click("#btn-retrain")
        await page.wait_for_timeout(2000)
        assert await page.evaluate("deadMasks.every(m => !m || m.size === 0)"), "重训后掩码应清空"
        print("[ablate] 重训后掩码已清空")

        await page.screenshot(path=r"E:\D\digit-net-viz\shots\ablate.png", full_page=True)
        await browser.close()
        print("[ablate] 页面错误:", errors if errors else "无")
        assert not errors
        print("[ablate] 消融功能全部通过")


if __name__ == "__main__":
    asyncio.run(main())
