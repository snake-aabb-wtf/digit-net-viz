"""消融功能验证：禁用/恢复神经元，数值断言在冻结权重快照内自洽比较（免疫训练 tick）。"""
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

        # ---- 数值核心：全部在冻结权重快照内完成，tick 不会干扰 ----
        num = await page.evaluate(
            """() => {
              const input = inputVec;
              const frozen = W.map(w => w.slice());   // 冻结权重
              const run = (fn) => { const old = W; W = frozen; try { return fn(); } finally { W = old; } };
              const snap = () => { acts = forward(input); probs = acts[acts.length - 1]; };
              const out = {};
              run(() => {
                deadMasks.forEach(m => m && m.clear());
                snap();
                out.base = Math.max(...probs);
                out.baseTop = probs.indexOf(Math.max(...probs)) + 1;
                out.strongest = acts[1].indexOf(Math.max(...acts[1]));
                out.baseP5 = probs[4];

                deadMasks[1].add(out.strongest);           // 单神经元消融
                snap();
                out.offAct = acts[1][out.strongest];
                out.offTop = probs.indexOf(Math.max(...probs)) + 1;
                out.off = Math.max(...probs);

                deadMasks[1].clear();                       // 恢复
                snap();
                out.re = Math.max(...probs);
                out.reP5 = probs[4];

                [...acts[1].keys()].sort((a, b) => acts[1][b] - acts[1][a]).slice(0, 30)
                    .forEach(j => deadMasks[1].add(j));     // 批量消融
                snap();
                out.batch = Math.max(...probs);
                out.batchTop = probs.indexOf(Math.max(...probs)) + 1;
                deadMasks[1].clear();
                snap();
              });
              return out;
            }"""
        )
        print(f"[ablate] 基线: top={num['baseTop']} 置信度={num['base']:.4f} p(5)={num['baseP5']:.4f} 最强H1=#{num['strongest']}")
        print(f"[ablate] 单消融: act={num['offAct']:.4f} top={num['offTop']} 置信度={num['off']:.4f}")
        print(f"[ablate] 恢复: 置信度={num['re']:.4f} p(5)={num['reP5']:.4f}")
        print(f"[ablate] 批量消融30: top={num['batchTop']} 置信度={num['batch']:.4f}")
        assert abs(num["offAct"]) < 1e-6, "被禁用神经元的激活应为 0"
        assert abs(num["re"] - num["base"]) < 1e-6, "恢复后输出应与基线完全一致"
        assert num["batch"] < num["base"] - 0.05 or num["batchTop"] != num["baseTop"], "批量消融应明显改变输出"

        # ---- UI 链路：点最强神经元 → 卡片与按钮状态（与权重版本无关） ----
        target = num["strongest"]
        await page.evaluate(
            f"""() => {{ const p = nodePos(1, {target}); const c = document.getElementById('net');
            const r = c.getBoundingClientRect();
            const sx = (p.x * view.k + view.tx) * r.width / netW;
            const sy = (p.y * view.k + view.ty) * r.height / netH;
            c.dispatchEvent(new MouseEvent('click',{{clientX:r.left+sx, clientY:r.top+sy, bubbles:true}})); }}"""
        )
        name = await page.evaluate("document.getElementById('rf-name').textContent")
        assert f"神经元 {target}" in name, f"点击应选中神经元 {target}，实际: {name}"
        await page.click("#rf-toggle")
        btn_txt = await page.evaluate("document.getElementById('rf-toggle').textContent")
        assert btn_txt == "重新启用此神经元"
        enabled = await page.evaluate("!document.getElementById('btn-enable-all').disabled")
        assert enabled, "有禁用时「启用所有神经元」应可用"
        print(f"[ablate] UI: 卡片 {name} · 按钮「{btn_txt}」 · 启用按钮可用: {enabled}")

        await page.click("#btn-enable-all")
        after = await page.evaluate(
            "({d: document.getElementById('btn-enable-all').disabled, t: document.getElementById('btn-enable-all').textContent})"
        )
        assert after["d"] and after["t"] == "启用所有神经元", "全部启用后按钮应置灰复原"
        print("[ablate] 启用所有神经元后按钮已复位")

        # ---- 重训清空掩码 ----
        await page.evaluate("deadMasks[2].add(3); deadMasks[2].add(4); updateEnableAllBtn();")
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
