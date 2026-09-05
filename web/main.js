"use strict";
/* 手写数字 · 神经网络观测台 —— 前端
   职责：WS 收权重快照 → 本地前向推理 → Canvas 渲染网络/热图/曲线/条形图 */

const $ = (id) => document.getElementById(id);
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

const CYAN = [56, 225, 212];
const CORAL = [255, 109, 109];
const AMBER = [255, 180, 84];
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

const MONO = '"Cascadia Mono", Consolas, monospace';
const WAVE_LAYER_MS = 170; // 波前到达相邻层的时间
const WAVE_NODE_MS = 240;  // 单层节点点亮时长

/* ---------------- 全局状态 ---------------- */
let arch = [784, 128, 64, 32, 9];
let W = [];            // Float32Array[] 权重（行主序 fin×fout）
let B = [];            // Float32Array[] 偏置
let inputVec = null;   // Float32Array(784) 最近一次画板输入
let acts = null;       // 各层激活 [a0..a4]
let probs = new Float32Array(9);
let animT0 = 0;        // 传播动画起点（0 = 无动画）
let drawing = false;
let liveFrame = false; // 绘制中的逐帧推理节流

let step = 0, epoch = 0, maxEpoch = 0;
let lossH = [], accH = [];
let weightsDirty = false, lastEdgeBuild = 0;
let edges = [];        // 每层 [[{s,t}]] 选中连线的源/目标下标
let edgeScale = [];    // 每层 alpha 归一化系数
let sourceName = "";
let doneFlag = false;
let retraining = false;
const deadMasks = [null, new Set(), new Set(), new Set()]; // 手动禁用的隐藏神经元（层→下标集合）
let rfTarget = null; // 感受野卡片当前指向的神经元 [l, j]
let hlEdges = [];       // 选中神经元的路径高亮：每层 Set("i-j")
let hlNodes = new Set(); // 路径经过的节点 "l:j"（l≥1）
let hlPixels = new Set(); // 路径经过的输入像素下标（0..783）
let view = { k: 1, tx: 0, ty: 0 }; // 画布视图：缩放系数 + 平移（逻辑像素）
let panState = null;    // 拖拽平移进行中的状态
let suppressClick = false; // 拖拽结束后吞掉紧随的 click

/* ---------------- 工具 ---------------- */
function f32fromB64(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

function forward(x) {
  const out = [x];
  let a = x;
  for (let l = 0; l < W.length; l++) {
    const fin = arch[l], fout = arch[l + 1], w = W[l], b = B[l];
    const y = new Float32Array(fout);
    if (l === W.length - 1) {
      let mx = -Infinity;
      for (let j = 0; j < fout; j++) {
        let s = b[j];
        for (let i = 0; i < fin; i++) s += a[i] * w[i * fout + j];
        y[j] = s;
        if (s > mx) mx = s;
      }
      let se = 0;
      for (let j = 0; j < fout; j++) { y[j] = Math.exp(y[j] - mx); se += y[j]; }
      for (let j = 0; j < fout; j++) y[j] /= se;
    } else {
      const dead = deadMasks[l + 1] || null; // 第 l 层矩阵算出第 l+1 层激活，掩码按层号 l+1 取
      for (let j = 0; j < fout; j++) {
        if (dead && dead.has(j)) continue; // 消融：该神经元输出强制为 0
        let s = b[j];
        for (let i = 0; i < fin; i++) s += a[i] * w[i * fout + j];
        y[j] = s > 0 ? s : 0;
      }
    }
    out.push(y);
    a = y;
  }
  return out;
}

/* ---------------- WebSocket ---------------- */
let ws = null;

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { mode = "live"; setLamp("training", "已连接 · 等待训练快照"); };
  ws.onclose = () => {
    if (mode === "live") { setLamp("err", "连接断开 · 2s 后重连"); setTimeout(connect, 2000); return; }
    if (mode === "static") return; // 静态模式不再碰 WS
    /* 从未连上（如 GitHub Pages 上没有 /ws）→ 尝试静态权重 */
    tryStatic().then((ok) => {
      if (ok) enterStatic();
      else { setLamp("err", "无服务 · 2s 后重试"); setTimeout(connect, 2000); }
    });
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "init") onInit(m);
    else if (m.type === "tick") onTick(m);
  };
}

/* ---------------- 静态部署模式（GitHub Pages） ---------------- */
let mode = "boot"; // boot | live | static
let staticInfo = null;
let timelineCache = null;
let replayTimer = null;

async function tryStatic() {
  try {
    const r = await fetch("weights.json", { cache: "no-cache" });
    if (!r.ok) return false;
    staticInfo = await r.json();
    return true;
  } catch {
    return false;
  }
}

function enterStatic() {
  mode = "static";
  arch = staticInfo.arch;
  sourceName = staticInfo.source;
  maxEpoch = staticInfo.epochs || 0;
  step = staticInfo.steps || 0;
  epoch = maxEpoch;
  $("net-sub").textContent = arch.join(" → ");
  $("data-info").innerHTML =
    `<b>数据集</b> ${sourceName === "mnist" ? "MNIST 在线子集（数字 1–9）" : "字体合成数据（兜底）"}<br>` +
    `<b>训练方式</b> GitHub Actions · 静态部署<br>` +
    `<b>训练样本</b> ${(staticInfo.trainN || 0).toLocaleString()} · <b>测试</b> ${(staticInfo.testN || 0).toLocaleString()}<br>` +
    `<b>最终准确率</b> ${(staticInfo.acc * 100).toFixed(1)}%` +
    (staticInfo.repo ? `<br><a href="https://github.com/${staticInfo.repo}/actions/workflows/deploy.yml" target="_blank" rel="noreferrer">在 Actions 上重新训练 ↗</a>` : "");
  buildBars(arch[arch.length - 1]);
  W = staticInfo.W.map(f32fromB64);
  B = staticInfo.b.map(f32fromB64);
  buildEdges();
  $("ro-step").textContent = step;
  $("ro-epoch").textContent = `${epoch}/${maxEpoch}`;
  $("ro-loss").textContent = staticInfo.loss.toFixed(3);
  $("ro-acc").textContent = (staticInfo.acc * 100).toFixed(1) + "%";
  lossH = [staticInfo.loss0, staticInfo.loss];
  accH = [staticInfo.acc0, staticInfo.acc];
  drawChart();
  setLamp("done", "静态部署模式 · CI 训练权重已加载");
  $("btn-retrain").textContent = "回放训练";
}

function q8fromB64(b64, s) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const i8 = new Int8Array(u8.buffer);
  const f = new Float32Array(i8.length);
  for (let i = 0; i < f.length; i++) f[i] = i8[i] * s;
  return f;
}

function applyReplayFrame(f) {
  W = f.Wq.map((o) => q8fromB64(o.d, o.s));
  B = f.b.map(f32fromB64);
  step = f.step; epoch = f.epoch;
  $("ro-step").textContent = step;
  $("ro-epoch").textContent = `${epoch}/${maxEpoch}`;
  $("ro-loss").textContent = f.loss == null ? "—" : f.loss.toFixed(3);
  $("ro-acc").textContent = f.acc == null ? "—" : (f.acc * 100).toFixed(1) + "%";
  if (f.loss != null) lossH.push(f.loss);
  if (f.acc != null) accH.push(f.acc);
  weightsDirty = true;
  if (inputVec) { acts = forward(inputVec); probs = acts[acts.length - 1]; updateBars(); }
  drawChart();
}

async function replayTraining() {
  if (mode !== "static" || replayTimer) return;
  const btn = $("btn-retrain");
  try {
    if (!timelineCache) {
      btn.disabled = true;
      btn.textContent = "加载时间线 …";
      const r = await fetch("timeline.json", { cache: "no-cache" });
      if (!r.ok) throw new Error("timeline 缺失");
      timelineCache = await r.json();
    }
  } catch {
    btn.disabled = false;
    btn.textContent = "回放训练（时间线缺失）";
    return;
  }
  /* 新网络从干净状态开始 */
  for (const m of deadMasks) if (m) m.clear();
  updateEnableAllBtn();
  $("rf-card").classList.remove("show");
  rfTarget = null;
  computeHighlight();
  const frames = timelineCache.frames;
  lossH = []; accH = [];
  btn.disabled = true;
  setLamp("training", "回放训练 …");
  await new Promise((resolve) => {
    let i = 0;
    replayTimer = setInterval(() => {
      applyReplayFrame(frames[i]);
      i++;
      btn.textContent = `回放中 ${i}/${frames.length}`;
      if (i >= frames.length) { clearInterval(replayTimer); replayTimer = null; resolve(); }
    }, timelineCache.frameMs || 350);
  });
  btn.disabled = false;
  btn.textContent = "回放训练";
  setLamp("done", "回放完成 · 试试画个数字");
}

function onInit(m) {
  arch = m.arch;
  sourceName = m.source;
  maxEpoch = m.epochs;
  $("net-sub").textContent = arch.join(" → ");
  $("data-info").innerHTML =
    `<b>数据集</b> ${m.source === "mnist" ? "MNIST 在线子集（数字 1–9）" : "字体合成数据（MNIST 不可用的兜底）"}<br>` +
    `<b>训练样本</b> ${m.trainN.toLocaleString()}<br>` +
    `<b>测试样本</b> ${m.testN.toLocaleString()}<br>` +
    `<b>训练轮次</b> ${m.epochs}`;
  buildBars(m.arch[4]);
  if (m.snapshot) onTick(m.snapshot, true);
}

function onTick(m, silent) {
  W = m.W.map(f32fromB64);
  B = m.b.map(f32fromB64);
  step = m.step; epoch = m.epoch;
  if (m.loss != null) lossH.push(m.loss);
  if (m.acc != null) accH.push(m.acc);
  if (lossH.length > 400) lossH.shift();
  if (accH.length > 400) accH.shift();
  doneFlag = m.done;
  weightsDirty = true;

  $("ro-step").textContent = step;
  $("ro-epoch").textContent = `${epoch}/${maxEpoch}`;
  $("ro-loss").textContent = m.loss == null ? "—" : m.loss.toFixed(3);
  $("ro-acc").textContent = m.acc == null ? "—" : (m.acc * 100).toFixed(1) + "%";

  if (doneFlag) setLamp("done", "训练完成 · 画个数字试试");
  else if (retraining) { retraining = false; enableRetrain(); setLamp("training", "重新训练中 …"); }
  else if (!silent) setLamp("training", `训练中 · 每 0.7s 推送权重`);

  if (inputVec) { acts = forward(inputVec); probs = acts[acts.length - 1]; updateBars(); }
  drawChart();
}

function setLamp(state, text) {
  const dot = $("lamp-dot");
  dot.className = state;
  $("lamp-text").textContent = text;
}

/* ---------------- 输出条形图 ---------------- */
function buildBars(n) {
  const box = $("bars");
  box.innerHTML = "";
  for (let j = 0; j < n; j++) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<span class="bar-digit">${j + 1}</span>` +
      `<span class="bar-track"><span class="bar-fill"></span></span>` +
      `<span class="bar-pct">0%</span>`;
    box.appendChild(row);
  }
  probs = new Float32Array(n);
}

function updateBars() {
  const rows = $("bars").children;
  let top = 0;
  for (let j = 1; j < probs.length; j++) if (probs[j] > probs[top]) top = j;
  for (let j = 0; j < rows.length; j++) {
    const p = probs[j];
    rows[j].querySelector(".bar-fill").style.width = (p * 100).toFixed(1) + "%";
    rows[j].querySelector(".bar-pct").textContent = (p * 100).toFixed(0) + "%";
    rows[j].classList.toggle("top", j === top && p > 0.02);
  }
  const v = $("verdict");
  if (probs[top] > 0.02) {
    v.innerHTML = `识别为 <b>${top + 1}</b> · 置信度 <b>${(probs[top] * 100).toFixed(1)}%</b>`;
  } else {
    v.textContent = "画点什么看看 …";
  }
}

/* ---------------- 训练曲线 ---------------- */
function drawChart() {
  const cv = $("chart");
  const dpr = devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(91,110,140,0.16)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    ctx.beginPath();
    ctx.moveTo(0, (h / 4.5) * i);
    ctx.lineTo(w, (h / 4.5) * i);
    ctx.stroke();
  }

  const plot = (arr, color, fixed01) => {
    if (arr.length < 2) return;
    let lo = Infinity, hi = -Infinity;
    for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (fixed01) { lo = 0; hi = 1; }
    if (hi - lo < 1e-6) hi = lo + 1e-6;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = 4 + (i / (arr.length - 1)) * (w - 8);
      const y = 8 + (1 - (arr[i] - lo) / (hi - lo)) * (h - 16);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  plot(lossH, rgba(CYAN, 0.95), false);
  plot(accH, rgba(AMBER, 0.95), true);

  ctx.font = `9px ${MONO}`;
  ctx.fillStyle = "rgba(91,110,140,0.9)";
  if (accH.length) ctx.fillText(`acc ${ (accH[accH.length - 1] * 100).toFixed(1) }%`, w - 58, 12);
  if (lossH.length) ctx.fillText(`loss ${lossH[lossH.length - 1].toFixed(2)}`, 6, 12);
}

/* ---------------- 网络可视化 ---------------- */
const netCv = $("net");
let netW = 0, netH = 0;

function layoutNet() {
  netW = netCv.clientWidth;
  netH = netCv.clientHeight;
  if (!netW || !netH) return;
  const dpr = devicePixelRatio || 1;
  netCv.width = netW * dpr;
  netCv.height = netH * dpr;
  if (typeof clampView === "function") clampView();
}

function netCtx() {
  const dpr = devicePixelRatio || 1;
  const ctx = netCv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/* 列位置：输入热图方块 → 3 个隐层 → 输出列 */
function colX() {
  const gridSide = Math.min(netH - 60, netW * 0.24);
  const gx = 24, gRight = gx + gridSide;
  const outX = netW - 76;
  const span = outX - gRight;
  return {
    gridSide, gx, gy: (netH - gridSide) / 2,
    hiddenX: [gRight + span * 0.25, gRight + span * 0.5, gRight + span * 0.75],
    outX,
  };
}

function hiddenY(l, j) {
  const n = arch[l];
  const top = 26, bot = netH - 26;
  return top + ((j + 0.5) * (bot - top)) / n;
}

function nodePos(li, idx) {
  const c = colX();
  if (li === 0) {
    const cell = c.gridSide / 28;
    return { x: c.gx + (idx % 28 + 0.5) * cell, y: c.gy + (Math.floor(idx / 28) + 0.5) * cell };
  }
  if (li <= 3) return { x: c.hiddenX[li - 1], y: hiddenY(li, idx) };
  return { x: c.outX, y: hiddenY(4, idx) };
}

/* 选边：每个目标节点只保留 |w| 最大的前 k 条入边 */
function buildEdges() {
  const K = [6, 7, 7, 7];
  edges = []; edgeScale = [];
  for (let l = 0; l < W.length; l++) {
    const fin = arch[l], fout = arch[l + 1], w = W[l];
    const k = Math.min(K[l] || 6, fin);
    const list = [];
    for (let j = 0; j < fout; j++) {
      const best = [];
      for (let i = 0; i < fin; i++) {
        const v = Math.abs(w[i * fout + j]);
        if (best.length < k) {
          best.push([v, i]);
          best.sort((a, b) => a[0] - b[0]);
        } else if (v > best[0][0]) {
          best[0] = [v, i];
          let p = 0;
          while (p + 1 < k && best[p][0] > best[p + 1][0]) {
            [best[p], best[p + 1]] = [best[p + 1], best[p]];
            p++;
          }
        }
      }
      for (const [, i] of best) list.push([i, j]);
    }
    let mx = 1e-6;
    for (const [i, j] of list) mx = Math.max(mx, Math.abs(w[i * fout + j]));
    edges.push(list);
    edgeScale.push(mx);
  }
  weightsDirty = false;
  lastEdgeBuild = performance.now();
  if (rfTarget) computeHighlight(); // 权重更新后按新连线重算路径高亮
}

/* 路径追踪：从选中神经元沿强连接边子集做上游 + 下游双向 BFS，
   收集所有「经过它」的连线、节点与输入像素 */
function computeHighlight() {
  hlEdges = []; hlNodes = new Set(); hlPixels = new Set();
  if (!rfTarget || !edges.length || edges.length !== W.length) return;
  const [sl, sj] = rfTarget;
  for (let l = 0; l < W.length; l++) hlEdges.push(new Set());

  hlNodes.add(sl + ":" + sj);
  /* 上游：逐层向输入方向回溯（edges[l] 连接层 l → 层 l+1，按目标匹配） */
  let frontier = new Set([sj]);
  for (let l = sl - 1; l >= 0; l--) {
    const next = new Set();
    for (const [i, j] of edges[l]) {
      if (frontier.has(j)) {
        hlEdges[l].add(i + "-" + j);
        if (l > 0) { hlNodes.add(l + ":" + i); next.add(i); }
        else hlPixels.add(i);
      }
    }
    frontier = next;
  }
  /* 下游：逐层向输出方向追踪（按源匹配，输出层不再外扩） */
  frontier = new Set([sj]);
  for (let l = sl; l < edges.length; l++) {
    const fout = arch[l + 1], w = W[l];
    const next = new Set();
    const covered = new Set();
    for (const [i, j] of edges[l]) {
      if (frontier.has(i)) {
        hlEdges[l].add(i + "-" + j);
        covered.add(i);
        if (l + 1 <= 3) { hlNodes.add((l + 1) + ":" + j); next.add(j); }
      }
    }
    /* 兜底：神经元在稀疏子集里没有出边时，按 |w| 补前 3 条，保证路径走通 */
    for (const i of frontier) {
      if (covered.has(i)) continue;
      const cand = [];
      for (let j = 0; j < fout; j++) {
        const v = Math.abs(w[i * fout + j]);
        if (v > 0) cand.push([v, j]);
      }
      cand.sort((a, b) => b[0] - a[0]);
      for (let t = 0; t < Math.min(3, cand.length); t++) {
        const j = cand[t][1];
        hlEdges[l].add(i + "-" + j);
        if (l + 1 <= 3) { hlNodes.add((l + 1) + ":" + j); next.add(j); }
      }
    }
    frontier = next;
  }
}

function drawNet(now) {
  if (!netW) layoutNet();
  const ctx = netCtx();
  ctx.clearRect(0, 0, netW, netH);

  if (!W.length) {
    ctx.font = `11px ${MONO}`;
    ctx.fillStyle = "rgba(91,110,140,0.8)";
    ctx.textAlign = "center";
    ctx.fillText("等待训练服务推送权重 …", netW / 2, netH / 2);
    return;
  }
  if (weightsDirty && now - lastEdgeBuild > 500) buildEdges();

  /* 视图变换：世界坐标（布局坐标）→ 屏幕 */
  ctx.save();
  ctx.translate(view.tx, view.ty);
  ctx.scale(view.k, view.k);

  const c = colX();
  const wavePos = animT0 ? (now - animT0) / WAVE_LAYER_MS : Infinity;
  const hlActive = !!rfTarget && hlEdges.length === edges.length;

  /* --- 连线 --- */
  for (let l = 0; l < edges.length; l++) {
    const fin = arch[l], fout = arch[l + 1], w = W[l];
    const passing = wavePos >= l && wavePos < l + 1.15; // 波前正穿过这层
    const boost = passing ? Math.max(0, 1 - Math.abs(wavePos - (l + 0.5)) / 0.9) : 0;
    const hlSet = hlActive ? hlEdges[l] : null;
    for (const [i, j] of edges[l]) {
      const wv = w[i * fout + j];
      const mag = Math.abs(wv) / edgeScale[l];
      const isHl = hlSet && hlSet.has(i + "-" + j);
      let alpha = 0.04 + mag * mag * 0.5 + boost * mag * 0.45;
      if (isHl) alpha = Math.max(alpha, 0.3 + mag * 0.65); // 路径线抬亮
      else if (hlActive) alpha *= 0.3;                     // 非路径线淡化
      if (!isHl) {
        if (deadMasks[l] && deadMasks[l].has(i)) alpha *= 0.08;      // 源被禁用
        if (deadMasks[l + 1] && deadMasks[l + 1].has(j)) alpha *= 0.08; // 目标被禁用
      }
      if (alpha < 0.02) continue;
      if (alpha > 1) alpha = 1;
      const p1 = nodePos(l, i), p2 = nodePos(l + 1, j);
      ctx.strokeStyle = rgba(wv >= 0 ? CYAN : CORAL, alpha);
      ctx.lineWidth = isHl ? 1.1 + mag * 1.7 + boost * 0.5 : 0.6 + mag * 1.1 + boost * 0.8;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }

  /* --- 输入热图 28×28 --- */
  {
    const cell = c.gridSide / 28;
    ctx.fillStyle = "rgba(91,110,140,0.25)";
    ctx.strokeRect(c.gx - 0.5, c.gy - 0.5, c.gridSide + 1, c.gridSide + 1);
    for (let i = 0; i < 784; i++) {
      const v = inputVec ? inputVec[i] : 0;
      if (v > 0.04) {
        let a = Math.min(1, v);
        if (hlActive && !hlPixels.has(i)) a *= 0.3; // 非路径像素淡化，路径像素保持点亮
        ctx.fillStyle = rgba([225, 250, 246], a);
        ctx.fillRect(c.gx + (i % 28) * cell, c.gy + Math.floor(i / 28) * cell, cell + 0.4, cell + 0.4);
      }
    }
    ctx.font = `9px ${MONO}`;
    ctx.fillStyle = "rgba(91,110,140,0.85)";
    ctx.textAlign = "center";
    ctx.fillText("28×28", c.gx + c.gridSide / 2, c.gy + c.gridSide + 14);
  }

  /* --- 隐层节点 --- */
  for (let l = 1; l <= 3; l++) {
    const n = arch[l];
    const x = c.hiddenX[l - 1];
    const r = Math.min(8, (netH - 52) / n * 0.34);
    const light = animT0 ? clamp01((now - animT0 - l * WAVE_LAYER_MS) / WAVE_NODE_MS) : 1;
    const dead = deadMasks[l];
    for (let j = 0; j < n; j++) {
      const a = acts ? acts[l][j] : 0;
      const glow = a * light;
      const y = hiddenY(l, j);
      const isSel = rfTarget && rfTarget[0] === l && rfTarget[1] === j;
      const isHlNode = hlActive && hlNodes.has(l + ":" + j);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 6.2832);
      if (dead && dead.has(j)) {
        /* 被禁用：灰色暗斑 + 虚线环 */
        ctx.fillStyle = "rgba(91,110,140,0.10)";
        ctx.fill();
        ctx.strokeStyle = "rgba(91,110,140,0.55)";
        ctx.lineWidth = 0.8;
        ctx.setLineDash([2, 2]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(x - r * 0.45, y - r * 0.45);
        ctx.lineTo(x + r * 0.45, y + r * 0.45);
        ctx.strokeStyle = "rgba(255,109,109,0.55)";
        ctx.lineWidth = 1;
        ctx.stroke();
        if (isSel) { // 选中已禁用的神经元：补琥珀环便于辨认
          ctx.beginPath();
          ctx.arc(x, y, r + 3, 0, 6.2832);
          ctx.strokeStyle = rgba(AMBER, 0.85);
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }
        continue;
      }
      ctx.fillStyle = rgba(CYAN, 0.05 + glow * 0.85);
      ctx.fill();
      if (isSel) {
        ctx.strokeStyle = rgba(AMBER, 0.95);
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, r + 3, 0, 6.2832);
        ctx.strokeStyle = rgba(AMBER, 0.4);
        ctx.lineWidth = 1;
        ctx.stroke();
      } else if (isHlNode) {
        ctx.strokeStyle = "rgba(233,245,242,0.85)";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      } else {
        ctx.strokeStyle = rgba(CYAN, 0.28 + glow * 0.6);
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }
    ctx.font = `9px ${MONO}`;
    ctx.fillStyle = "rgba(91,110,140,0.85)";
    ctx.textAlign = "center";
    ctx.fillText(String(n), x, netH - 8);
  }

  /* --- 输出节点 + 数字 --- */
  {
    const n = arch[4];
    const r = 10;
    let top = 0;
    for (let j = 1; j < n; j++) if (probs[j] > probs[top]) top = j;
    const light = animT0 ? clamp01((now - animT0 - 4 * WAVE_LAYER_MS) / WAVE_NODE_MS) : 1;
    ctx.font = `13px ${MONO}`;
    for (let j = 0; j < n; j++) {
      const p = acts ? acts[4][j] : 0;
      const glow = p * light;
      const y = hiddenY(4, j);
      ctx.beginPath();
      ctx.arc(c.outX, y, r, 0, 6.2832);
      ctx.fillStyle = rgba(CYAN, 0.05 + glow * 0.9);
      ctx.fill();
      const isHlOut = hlActive && hlNodes.has("4:" + j);
      if (j === top && glow > 0.15) {
        ctx.strokeStyle = rgba(CYAN, 0.95);
        ctx.lineWidth = 1.6;
      } else if (isHlOut) {
        ctx.strokeStyle = "rgba(233,245,242,0.85)";
        ctx.lineWidth = 1.2;
      } else {
        ctx.strokeStyle = rgba(CYAN, 0.3 + glow * 0.5);
        ctx.lineWidth = 0.8;
      }
      ctx.stroke();
      ctx.fillStyle = j === top && glow > 0.15 ? rgba(CYAN, 1) : "rgba(91,110,140,0.9)";
      ctx.textAlign = "left";
      ctx.fillText(String(j + 1), c.outX + r + 9, y + 4.5);
    }
  }

  /* --- 空输入提示 --- */
  if (!inputVec) {
    ctx.font = `10px ${MONO}`;
    ctx.fillStyle = "rgba(91,110,140,0.6)";
    ctx.textAlign = "center";
    ctx.fillText("画一个数字，信号将从这里出发", c.gx + c.gridSide / 2, c.gy - 12);
  }

  ctx.restore();
}

/* ---------------- 感受野（点击隐藏节点） ---------------- */
function columnOf(w, fin, fout, j) {
  const v = new Float32Array(fin);
  for (let i = 0; i < fin; i++) v[i] = w[i * fout + j];
  return v;
}

function receptiveField(l, j) {
  /* 输入空间感受野：rf = W0 · (W1 · (… · e_j))，符号保留 */
  let v = columnOf(W[l - 1], arch[l - 1], arch[l], j);
  for (let k = l - 2; k >= 0; k--) {
    const w = W[k], fin = arch[k], fout = arch[k + 1];
    const nv = new Float32Array(fin);
    for (let i = 0; i < fin; i++) {
      let s = 0;
      for (let t = 0; t < fout; t++) s += w[i * fout + t] * v[t];
      nv[i] = s;
    }
    v = nv;
  }
  return v;
}

function showRF(l, j) {
  const v = receptiveField(l, j);
  let mx = 1e-6;
  for (let i = 0; i < v.length; i++) mx = Math.max(mx, Math.abs(v[i]));
  const cv = $("rf-canvas");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, 28, 28);
  const img = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const t = v[i] / mx;
    const c = t >= 0 ? CYAN : CORAL;
    const a = Math.min(1, Math.abs(t)) * 255;
    img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2];
    img.data[i * 4 + 3] = a;
  }
  ctx.putImageData(img, 0, 0);
  rfTarget = [l, j];
  $("rf-name").textContent = `H${l} · 神经元 ${j}` + (deadMasks[l] && deadMasks[l].has(j) ? "（已禁用）" : "");
  const tbtn = $("rf-toggle");
  const isDead = deadMasks[l].has(j);
  tbtn.textContent = isDead ? "重新启用此神经元" : "禁用此神经元";
  tbtn.classList.toggle("on", isDead);
  $("rf-card").classList.add("show");
}

function updateEnableAllBtn() {
  const total = deadMasks.reduce((s, m) => s + (m ? m.size : 0), 0);
  const btn = $("btn-enable-all");
  btn.disabled = total === 0;
  btn.textContent = total === 0 ? "启用所有神经元" : `启用所有神经元（${total} 已禁用）`;
}

$("rf-toggle").addEventListener("click", () => {
  if (!rfTarget) return;
  const [l, j] = rfTarget;
  const set = deadMasks[l];
  if (set.has(j)) set.delete(j); else set.add(j);
  if (inputVec) { acts = forward(inputVec); probs = acts[acts.length - 1]; updateBars(); }
  showRF(l, j); // 刷新卡片文案（按钮状态）
  computeHighlight();
  updateEnableAllBtn();
});

$("btn-enable-all").addEventListener("click", () => {
  for (const m of deadMasks) if (m) m.clear();
  if (inputVec) { acts = forward(inputVec); probs = acts[acts.length - 1]; updateBars(); }
  if (rfTarget) showRF(rfTarget[0], rfTarget[1]);
  computeHighlight();
  updateEnableAllBtn();
});

netCv.addEventListener("click", (e) => {
  if (suppressClick) { suppressClick = false; return; } // 拖拽平移不算点击
  const rect = netCv.getBoundingClientRect();
  /* 屏幕坐标 → 世界坐标（除以视图变换） */
  const sx = ((e.clientX - rect.left) * (netW / rect.width) - view.tx) / view.k;
  const sy = ((e.clientY - rect.top) * (netH / rect.height) - view.ty) / view.k;
  /* 在全部隐层里找距离最近的节点（密集层先到先得会抓错邻居） */
  let best = null, bestD = Infinity;
  for (let l = 1; l <= 3; l++) {
    const n = arch[l];
    const r = Math.min(8, (netH - 52) / n * 0.34) + 7;
    const x = colX().hiddenX[l - 1];
    for (let j = 0; j < n; j++) {
      const y = hiddenY(l, j);
      const d = (x - sx) ** 2 + (y - sy) ** 2;
      if (d < r * r && d < bestD) { best = [l, j]; bestD = d; }
    }
  }
  if (best) { showRF(best[0], best[1]); computeHighlight(); }
  else { $("rf-card").classList.remove("show"); rfTarget = null; computeHighlight(); }
});

/* ---------------- 视图：滚轮缩放 + 拖拽平移 + 双击复位 ---------------- */
const VIEW_MIN = 0.5, VIEW_MAX = 8;

function clampView() {
  view.k = Math.min(VIEW_MAX, Math.max(VIEW_MIN, view.k));
  /* 平移上限：内容边到画布边之间，避免把网络甩出视野 */
  const contentW = netW, contentH = netH;
  const maxX = (contentW * (view.k - 1)) / 2 + contentW * 0.6;
  const maxY = (contentH * (view.k - 1)) / 2 + contentH * 0.6;
  view.tx = Math.min(maxX, Math.max(-maxX, view.tx));
  view.ty = Math.min(maxY, Math.max(-maxY, view.ty));
}

function screenToWorld(px, py) {
  return { x: (px - view.tx) / view.k, y: (py - view.ty) / view.k };
}

function zoomAt(px, py, factor) {
  const before = screenToWorld(px, py);
  view.k = Math.min(VIEW_MAX, Math.max(VIEW_MIN, view.k * factor));
  /* 让缩放前光标下的世界坐标保持在光标下（锚点缩放） */
  view.tx = px - before.x * view.k;
  view.ty = py - before.y * view.k;
  clampView();
}

netCv.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (!netW) return;
  const rect = netCv.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (netW / rect.width);
  const py = (e.clientY - rect.top) * (netH / rect.height);
  /* deltaMode=1（行滚动）时按 16px/行折算 */
  const unit = e.deltaMode === 1 ? 16 : 1;
  const factor = Math.exp(-e.deltaY * unit * 0.0022);
  zoomAt(px, py, factor);
  netCv.style.cursor = view.k > 1 ? "grab" : "";
}, { passive: false });

netCv.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || view.k <= 1) return; // 仅放大后允许拖拽平移
  panState = {
    id: e.pointerId,
    sx: e.clientX, sy: e.clientY,
    tx0: view.tx, ty0: view.ty,
    moved: false,
  };
  netCv.setPointerCapture(e.pointerId);
  netCv.style.cursor = "grabbing";
});

netCv.addEventListener("pointermove", (e) => {
  if (!panState || e.pointerId !== panState.id) return;
  const dx = e.clientX - panState.sx, dy = e.clientY - panState.sy;
  if (!panState.moved && Math.hypot(dx, dy) > 3) panState.moved = true;
  if (panState.moved) {
    view.tx = panState.tx0 + dx;
    view.ty = panState.ty0 + dy;
    clampView();
  }
});

function endPan(e) {
  if (!panState || e.pointerId !== panState.id) return;
  if (panState.moved) suppressClick = true; // 平移结束，吞掉随后的 click
  panState = null;
  netCv.style.cursor = view.k > 1 ? "grab" : "";
}
netCv.addEventListener("pointerup", endPan);
netCv.addEventListener("pointercancel", endPan);

netCv.addEventListener("dblclick", () => {
  view = { k: 1, tx: 0, ty: 0 };
});

/* ---------------- 画板 ---------------- */
const paint = $("paint");
const pctx = paint.getContext("2d", { willReadFrequently: true });
pctx.lineCap = "round";
pctx.lineJoin = "round";
pctx.strokeStyle = "#e9f5f2";
pctx.lineWidth = 21;

const t28 = document.createElement("canvas"); t28.width = 28; t28.height = 28;
const t2 = document.createElement("canvas"); t2.width = 28; t2.height = 28;

function paintXY(e) {
  const r = paint.getBoundingClientRect();
  return { x: (e.clientX - r.left) * (280 / r.width), y: (e.clientY - r.top) * (280 / r.height) };
}

paint.addEventListener("pointerdown", (e) => {
  drawing = true;
  paint.setPointerCapture(e.pointerId);
  const p = paintXY(e);
  pctx.beginPath();
  pctx.moveTo(p.x, p.y);
  pctx.lineTo(p.x + 0.01, p.y);
  pctx.stroke();
  $("paint-hint").classList.add("off");
  requestLiveInfer();
});

paint.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = paintXY(e);
  pctx.lineTo(p.x, p.y);
  pctx.stroke();
  requestLiveInfer();
});

function endStroke() {
  if (!drawing) return;
  drawing = false;
  infer(true);
}
paint.addEventListener("pointerup", endStroke);
paint.addEventListener("pointercancel", endStroke);

function requestLiveInfer() {
  if (liveFrame) return;
  liveFrame = true;
  requestAnimationFrame(() => { liveFrame = false; infer(false); });
}

function clearPaint() {
  pctx.clearRect(0, 0, 280, 280);
  inputVec = null;
  acts = null;
  probs.fill(0);
  animT0 = 0;
  updateBars();
  $("paint-hint").classList.remove("off");
  $("verdict").textContent = "等待输入 …";
}

$("btn-clear").addEventListener("click", clearPaint);

$("btn-retrain").addEventListener("click", () => {
  if (mode === "static") { replayTraining(); return; }
  if (!ws || ws.readyState !== 1) return;
  retraining = true;
  ws.send("retrain");
  for (const m of deadMasks) if (m) m.clear(); // 新网络从干净状态开始
  updateEnableAllBtn();
  $("rf-card").classList.remove("show");
  rfTarget = null;
  computeHighlight();
  lossH = []; accH = [];
  step = 0; epoch = 0;
  $("ro-step").textContent = "0";
  $("ro-epoch").textContent = `0/${maxEpoch}`;
  $("ro-loss").textContent = "—";
  $("ro-acc").textContent = "—";
  drawChart();
  setLamp("training", "权重重置 · 重新训练中 …");
  const btn = $("btn-retrain");
  btn.disabled = true;
  setTimeout(enableRetrain, 4000);
});

function enableRetrain() { $("btn-retrain").disabled = false; }

/* 画板 → 28×28 MNIST 规格（bbox 裁剪 + 等比缩放 + 质心居中） */
function preprocess() {
  const c28 = t28.getContext("2d", { willReadFrequently: true });
  c28.clearRect(0, 0, 28, 28);
  c28.drawImage(paint, 0, 0, 28, 28);
  const d = c28.getImageData(0, 0, 28, 28).data;
  let minX = 28, minY = 28, maxX = -1, maxY = -1;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      if (d[(y * 28 + x) * 4 + 3] > 30) {
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  const w = maxX - minX + 1, h = maxY - minY + 1;
  const s = 20 / Math.max(w, h);
  const sw = Math.max(1, Math.round(w * s)), sh = Math.max(1, Math.round(h * s));

  const c2 = t2.getContext("2d", { willReadFrequently: true });
  c2.clearRect(0, 0, 28, 28);
  c2.imageSmoothingEnabled = true;
  c2.imageSmoothingQuality = "high";
  c2.drawImage(paint, minX * 10, minY * 10, w * 10, h * 10, 0, 0, sw, sh);

  const d2 = c2.getImageData(0, 0, sw, sh).data;
  let sx = 0, sy = 0, cnt = 0;
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      if (d2[(y * sw + x) * 4 + 3] > 60) { sx += x; sy += y; cnt++; }
    }
  }
  if (!cnt) return null;
  const dx = Math.round(14 - sx / cnt), dy = Math.round(14 - sy / cnt);

  const t3 = document.createElement("canvas"); t3.width = 28; t3.height = 28;
  const c3 = t3.getContext("2d", { willReadFrequently: true });
  c3.drawImage(t2, 0, 0, sw, sh, dx, dy, sw, sh);
  const d3 = c3.getImageData(0, 0, 28, 28).data;
  const out = new Float32Array(784);
  let mx = 0;
  for (let i = 0; i < 784; i++) {
    out[i] = d3[i * 4 + 3] / 255;
    if (out[i] > mx) mx = out[i];
  }
  if (mx > 0 && mx < 1) for (let i = 0; i < 784; i++) out[i] = Math.min(1, out[i] / mx * 0.95);
  return out;
}

function infer(withWave) {
  if (!W.length) return;
  const v = preprocess();
  if (!v) { if (!drawing) clearPaint(); return; }
  inputVec = v;
  acts = forward(v);
  probs = acts[acts.length - 1];
  updateBars();
  if (withWave && !REDUCED) animT0 = performance.now();
}

/* ---------------- 主循环 ---------------- */
function frame(now) {
  drawNet(now);
  requestAnimationFrame(frame);
}

new ResizeObserver(layoutNet).observe($("net-panel"));
window.addEventListener("resize", () => { layoutNet(); drawChart(); });
layoutNet();
buildEdgesSafe();
connect();
requestAnimationFrame(frame);

function buildEdgesSafe() { /* 权重未到时仅复位标记 */ edges = []; edgeScale = []; }
