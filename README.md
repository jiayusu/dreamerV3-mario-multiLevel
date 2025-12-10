# dreamerV3-mario-multiLevel
应用机器学习之超级马力欧作业
> 关卡条件世界模型，让超级马力欧兄弟多关卡通用控制成为可能

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-实时跟踪-yellowgreen.svg)](https://wandb.ai)

## 🔍 为什么（Why）
传统 RL 在《超级马力欧兄弟》面临两大瓶颈：
1. **手工特征**——Liao 等（2012）需 39 维人工状态；
2. **单关过拟合**——CNN 策略难以迁移到 1-2、2-1 等新关。

我们将 **DreamerV3** 扩展为**关卡条件 RSSM**：
- 一个**共享世界模型**学习通用物理（跳跃、敌人、卷轴）；
- **关卡嵌入**让策略专精每关几何；
- **MCTS 蒸馏**在稀疏奖励（管道、深坑）中提升探索。

→ **端到端像素输入，样本高效，跨关通用。**

## ⚙️ 怎么做（How）
0. **混合 MDP 视角**  
  把每关当潜在上下文 L；世界模型学 $p(o_{t+1},r_t|h_t,z_t,a_t,L)$。

1. **因子化架构**  
  ```
  像素 → CNN → 256-D → RSSM(潜在动力学) → Decoder(重建)
                            ↓
                   Actor/Critic(+关卡emb) → 动作
  ```

2. **训练循环**  
  - **4×异步采集器**(CPU) 持续喂完整幕次  
  - **序列 PER** 按「绝对幕次回报」优先采样  
  - **想象**：50 步潜在 rollout → GAE → PPO 更新  
  - **MCTS 蒸馏**：30 模拟×5 步 → KL 损失注入 Actor（每批 2 样本）  
  - **W&B 实时**：loss、分关回报、缓冲大小一键可视化

3. **关键修复**（vs 原版）
  - 采集器显式 `device="cpu"` → 无死锁  
  - 所有 `action.long().to(h.device)` → 零设备/类型不匹配  
  - 内层 `steps < COLLECT_STEPS` 守卫 → 杜绝无限幕次 bug  
  - 统一 `.pth` + 配置字典 → 单文件部署

## 🚀 你能得到什么（What）
TODO：实验结论

*测试环境：*

## 📦 一行安装 & 运行
```bash
# 1. 克隆
git clone https://github.com/YOUR_NAME/DreamerV3-Mario-MultiLevel.git
cd DreamerV3-Mario-MultiLevel

# 2. 创建环境与依赖
conda env create -f environment.yml
conda activate mario38

# 3. 登录 W&B
wandb login

# 4. 开训
python train.py
```

## 📁 文件说明
| 文件 | 作用 |
|------|------|
| `train.py` | **单文件训练脚本** |
| `dreamer_mario_final.pt` | 已释出的 1-1 & 1-2  checkpoint |
| `README.md` | 本说明 |
| `requirements.txt` | PyPI 依赖列表 |


## 🤝 贡献
欢迎提 PR 或开 Issue！  
- 追加更多关卡（2-1、3-1 …）  
- 用 LVLM 替换 MCTS 做高层次规划  
- TorchScript / ONNX 导出，嵌入式马力欧

## 📬 联系
直接在 GitHub Issue 或 Discussion 留言，我会定期查看。
