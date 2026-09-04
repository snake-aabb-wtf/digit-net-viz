# 手写数字 · 神经网络观测台

在浏览器里看一个神经网络**现场学会**认你手写的 1–9。

- **后端**（`server/`）：FastAPI + numpy 手写 MLP（784→128→64→32→9），
  在 MNIST 子集（只保留数字 1–9）上用 SGD+Momentum 训练，
  每 0.7s 把权重快照 + loss/准确率通过 WebSocket 推给浏览器。
- **前端**（`web/`）：原生 JS + Canvas，零构建。拿到权重后在浏览器本地做推理，
  支持画板实时识别、权重连线、前向传播动画、隐藏层神经元感受野、训练曲线。

## 启动

```bash
cd E:\D\digit-net-viz
pip install -r requirements.txt
python -m server.app
```

然后打开 **http://127.0.0.1:8123**

## 首次运行

会自动从 S3 镜像下载 MNIST（约 12 MB）并缓存到 `data/`；
下载失败时自动退回「字体合成数据」训练，程序照样能跑。

## 玩法

1. 打开页面，观察右下角 loss 曲线下降、准确率爬升，中间网络连线逐渐成形
2. 在左侧画板写一个 1–9，松手 → 看信号逐层点亮传播，右侧给出概率
3. 点击中间任意隐藏层节点 → 弹出它学到的「输入空间感受野」
4. 点「重新训练」→ 权重从随机重新开始，再看一遍它学会的全过程

## 校验

`python verify.py`（需 `pip install playwright && playwright install chromium`）
会用无头浏览器自动画 1/3/7 并核对识别结果。
