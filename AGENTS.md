# AGENTS.md —— 给在本仓库工作的 AI 代理

## 项目是什么

「手写数字 · 神经网络观测台」：Python 后端训练一个 numpy 手写 MLP
（784→128→64→32→9，识别数字 1–9），通过 WebSocket 每 0.7s 推送权重快照；
原生 JS/Canvas 前端在浏览器本地推理并做全套可视化（训练曲线、传播动画、
感受野、神经元消融、路径高亮、缩放平移）。

## 硬约束（除非用户明确要求，否则不要违反）

- **后端只用 numpy + FastAPI**，不引入 torch/tf/sklearn。
- **前端零依赖**：原生 JS + Canvas，无 npm、无构建链、无框架、无 CDN 字体。
- UI 文案用简体中文；代码注释用中文；配色语义固定：
  青 `--cyan`=正权重/激励，珊瑚 `--coral`=负权重/抑制，琥珀 `--amber`=准确率/选中高亮。
- 服务端口 8123；WS 快照节拍 0.7s（`server/trainer.py: TICK_SECONDS`）。

## 目录

```
server/  data.py(数据+兜底) model.py(MLP) trainer.py(在线训练线程)
         train_export.py(无服务训练导出 weights.json + timeline.json)
         app.py(FastAPI)
web/     index.html style.css main.js（全部渲染与交互逻辑在 main.js）
.github/workflows/  ci.yml(push/PR 回归) deploy.yml(训练→导出→Pages 部署)
verify.py ablate.py zoom.py highlight.py probe.py selftest.py  # Playwright 测试
data/    MNIST .npy 缓存（gitignore，勿提交）
shots/   测试截图（gitignore）
dist/    train_export 产物（gitignore）
```

## 双运行模式（改 main.js 前先弄清当前模式）

- **live 模式**：WS 连上 `server.app`，每 0.7s 收 tick 快照（现状逻辑）。
- **static 模式**：WS 连不上（GitHub Pages 上 `/ws` 必 404）→ `tryStatic()`
  加载 `weights.json` 进入静态模式；「重新训练」按钮变为「回放训练」，
  按 `timeline.json` 逐帧应用 int8 量化权重（`applyReplayFrame`）。
  两个模式共用 forward/渲染/交互代码，新模式判断用全局 `mode` 变量。

## 改动后的标准验证流程

1. `node --check web/main.js`（语法关，必跑）
2. 确认服务在跑：`(Invoke-WebRequest http://127.0.0.1:8123/api/status).Content`
   （没跑就后台启动 `python -m server.app`）
3. 按改动范围跑对应 Playwright 脚本（见下表）；**控制台设
   `$env:PYTHONIOENCODING='utf-8'`**，否则中文输出乱码
4. 全量回归 = `verify.py` + `ablate.py` + `zoom.py` + `highlight.py` 全绿

| 测试 | 覆盖 |
|---|---|
| verify.py | 画 1/3/7 识别、感受野卡片、重训按钮 |
| ablate.py | 禁用/恢复/批量消融/重训清空掩码 |
| zoom.py | 滚轮缩放锚点、拖拽平移、双击复位、缩放态命中 |
| highlight.py | 选中神经元的全层路径高亮与取消 |
| probe.py | 像素级渲染体检（各画布区域非空） |
| selftest.py | 静态模式降级、回放训练、回放后识别（需先跑 train_export） |

## 架构要点（改 main.js 前必读）

- **坐标体系**：布局全部是「世界坐标」（`colX()/hiddenY()/nodePos()`）；
  渲染时套视图变换 `translate(view.tx, view.ty) + scale(view.k)`；
  命中测试必须用 `screenToWorld()` 反变换。canvas 尺寸一律
  `clientWidth/clientHeight × devicePixelRatio`（hidpi）。
- **W[l] 形状 (arch[l], arch[l+1]) 行主序**：元素 (i,j) 在 `i*fout + j`。
- **边子集是稀疏的**：`buildEdges()` 只保留每个目标节点的 top-k 入边，
  因此**按源找出边可能为空**——任何下游遍历/路径计算都必须带
  `|w|` 兜底（参考 `computeHighlight()` 的 fallback）。
- **WS tick 每 0.7s 全量替换 W/B**（约 1MB JSON），依赖 W 的派生缓存
  （如高亮、选边）要么在 `buildEdges()` 末尾重算，要么容忍 ≤0.5s 旧值。

## 已知的坑（都真实踩过）

1. **层号差一**：`forward()` 里第 `l` 个权重矩阵算出的是第 `l+1` 层激活；
   `deadMasks` 按层号 1..3 索引，所以在 forward 内取 `deadMasks[l+1]`。
2. **classic script 的全局**：main.js 顶层 `function` 声明挂 `window`，
   `let/const` 不挂但可直接按名访问 —— Playwright `evaluate` 里直接写
   变量名，别写 `window.W`。
3. **训练很快**：十几秒跑完 18 轮。测试可能落在「训练中」或「已完成」
   两种状态（别的测试还会点重训），断言不要写死 step/epoch。
4. MNIST 下载失败会自动兜底合成数据（`source: "synthetic"`），
   断言数据源要兼容两种取值。
5. Windows git 的 `LF will be replaced by CRLF` warning 属正常，忽略。
6. `data/`、`shots/`、`dist/` 已 gitignore；发现它们被暂存通常是 .gitignore
   行内注释失效（`#` 只在行首才是注释）。
7. **int8 量化 scale = max|a|/127**（`train_export.py: b64_i8`）。
   踩过的坑：把 max|a| 本身当 scale 会让权重放大 127 倍，ReLU 层侥幸存活，
   softmax 却 `Inf/Inf = NaN` —— 报「识别输出 NaN」先查量化缩放。
8. 静态站上 `/ws` 404 是**预期的降级信号**（console 会出现 WebSocket error），
   测试脚本要过滤掉它，别当失败。
9. `load_dataset()` 缓存命中与下载两条路径都要给 `source` 赋值
   （曾因缓存命中分支漏赋值 UnboundLocalError）。

## 提交规范

`main` 分支直接提交；一个功能一个 commit；提交信息为中文一句话，
格式如「新增 XX」「修复 YY」。
