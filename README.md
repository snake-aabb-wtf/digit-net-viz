# 手写数字 · 神经网络观测台 (Digit Observatory)

在浏览器里看一个神经网络**现场学会**认你手写的 1–9。

后端用 numpy 手写一个小型 MLP 并在 MNIST 子集上训练，通过 WebSocket 把权重快照
实时推给浏览器；前端零依赖（原生 JS + Canvas），在本地做推理并把一切可视化：
权重连线、信号传播动画、感受野、训练曲线，甚至让你亲手「切掉」某个神经元看它掉链子。

## 功能

- **训练全程可视化**：loss 曲线下降、测试准确率爬升、连线从混沌长出结构（每 0.7s 推一帧权重）
- **手写实时识别**：画板落笔即推理，松手回放信号逐层点亮的前向传播动画
- **感受野**：点击任意隐藏神经元，看它学到的「输入空间感受野」（青=兴奋输入，红=抑制输入）
- **神经元消融实验**：一键禁用/恢复任意隐藏神经元，观察冗余网络如何（或不会）崩溃
- **路径追踪**：选中神经元后，高亮全部经过它的连线（贯穿 784→128→64→32→9 与输入像素）
- **自由观察**：网络画布支持滚轮缩放（光标锚点）、拖拽平移、双击复位
- **重新训练**：一键从随机权重重看整个学习过程

## 架构

```
┌─────────────── server/ (Python) ───────────────┐      ┌────────── web/ (浏览器) ──────────┐
│ data.py    下载/缓存 MNIST（失败→字体合成兜底） │      │ main.js                           │
│ model.py   numpy MLP 784→128→64→32→9            │      │  ├─ WS 收权重快照（每 0.7s）       │
│ trainer.py 后台训练线程 + 快照发布              │ ─WS→ │  ├─ 本地前向推理（落笔零延迟）     │
│ app.py     FastAPI /ws /api/status + 静态托管   │      │  └─ Canvas 渲染网络/热图/曲线/条形 │
└────────────────────────────────────────────────┘      └───────────────────────────────────┘
```

网络：784 → 128 → 64 → 32 → 9（数字 1–9，MNIST 过滤掉 0），
ReLU 隐层 + Softmax 输出，SGD + Momentum（batch 64，18 轮，测试准确率 ~97%）。

## 快速开始

```bash
pip install -r requirements.txt
python -m server.app
# 打开 http://127.0.0.1:8123
```

首次运行会从 S3 镜像下载 MNIST（约 12 MB）并缓存到 `data/`（已 gitignore）；
下载失败时自动退回「字体合成数据」训练，程序照样能跑。

## 在线版（GitHub Pages + Actions）

仓库推送 `main` 后，`deploy.yml` 会在 **GitHub Actions 的机器上**完成训练，
导出 `weights.json`（最终权重）与 `timeline.json`（67 帧 int8 量化训练时间线），
自检通过后发布到 GitHub Pages —— 在线版无需任何本地服务：

> **https://snake-aabb-wtf.github.io/digit-net-viz/**

前端自动降级：探测不到 WebSocket（`/ws`）时进入**静态模式**，
加载 CI 训练好的权重；「重新训练」按钮变为「**回放训练**」，
按时间线逐帧回放从随机到学会的全过程（约 24 秒）。
也可在仓库 Actions 页手动触发 *Train & Deploy to Pages* 重新训练。

`ci.yml` 则在每次 push/PR 时自动跑四套 Playwright 回归（verify/ablate/zoom/highlight）。

手动在本地导出静态产物：

```bash
python -m server.train_export --out dist   # 生成 dist/weights.json + dist/timeline.json
```

## 交互指南

| 操作 | 效果 |
|---|---|
| 画板写 1–9 | 实时推理，松手回放传播动画 |
| 点击隐藏神经元 | 感受野卡片 + 高亮经过它的全部连线 |
| 卡片内「禁用此神经元」 | 该神经元输出归零（消融实验） |
| 「启用所有神经元」 | 一键恢复全部被禁用神经元 |
| 网络区滚轮 / 拖拽 / 双击 | 缩放 / 平移 / 复位视图 |
| 「重新训练」 | 权重从随机重置，重看学习全过程 |

## WebSocket 协议

- 服务端 → 客户端 `init`：`{arch, source, trainN, testN, epochs, snapshot}`
- 服务端 → 客户端 `tick`：`{seq, step, epoch, loss, acc, done, W[], b[]}`
  （W/b 为 base64 编码的 Float32 行主序数组，形状 `arch[l] × arch[l+1]`）
- 客户端 → 服务端：文本 `retrain`

## 项目结构

```
server/
  data.py      # MNIST 下载/解析/缓存 + 合成数据兜底
  model.py     # numpy MLP（前向/反向/精度）
  trainer.py   # 训练线程 + 快照发布（BATCH/EPOCHS/TICK_SECONDS 在此调）
  app.py       # FastAPI 入口、/ws、/api/status、静态托管
web/
  index.html   # 三栏观测台布局
  style.css    # 深色示波器风格（青=正权重 红=负权重 琥珀=准确率）
  main.js      # WS 客户端、推理、网络渲染、交互
verify.py      # 端到端：画 1/3/7 → 核对识别 → 感受野 → 重训
ablate.py      # 消融功能验证
zoom.py        # 缩放/平移交互验证
highlight.py   # 路径高亮验证
probe.py       # 像素级渲染体检
```

## 测试

需要 Playwright：`pip install playwright && playwright install chromium`，
并先启动服务（`python -m server.app`），然后：

```bash
python verify.py      # 画 1/3/7，断言识别正确、感受野与重训可用
python ablate.py      # 禁用/恢复神经元、批量消融、重训清空掩码
python zoom.py        # 滚轮缩放锚点、拖拽平移、双击复位、缩放态命中
python highlight.py   # 选中神经元的全层路径高亮
```

## License

[MIT](LICENSE)
