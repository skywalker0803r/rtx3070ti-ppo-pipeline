# RTX 3070 Ti PPO 強化學習管線

這是一個使用 PyTorch 與 Gymnasium 建立的輕量化 PPO 訓練專案，目標環境為
`BipedalWalker-v3`，並針對 8 GB NVIDIA RTX 3070 Ti 進行設定。

## 功能

- 使用 32 個平行環境收集 rollout
- Actor-Critic 網路使用 Orthogonal Initialization
- 支援 GAE（Generalized Advantage Estimation）與 PPO clipped objective
- 支援 CUDA、AMP 自動混合精度與可選的 `torch.compile()`
- 使用 TensorBoard 記錄 FPS、Loss、Reward 與 VRAM
- Reward 提升時自動儲存 `checkpoints/best_model.pt`
- 使用 `render.py` 開啟實際的 BipedalWalker 動畫

## 系統需求

- Linux
- Python 3.10 或更新版本
- 支援 CUDA 的 NVIDIA 驅動程式
- NVIDIA RTX 3070 Ti 或其他 CUDA GPU

完整 Python 套件清單請參考 [requirements.txt](requirements.txt)。

## 安裝

Debian/Ubuntu 若尚未安裝 Python 虛擬環境模組，先執行：

```bash
sudo apt install python3-venv
```

建立並啟用虛擬環境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## GPU 硬體檢查

開始訓練前，先確認 PyTorch 能辨識 RTX 3070 Ti：

```bash
python benchmark.py
```

此指令會顯示 GPU 型號、VRAM 容量、矩陣乘法效能，以及目前已配置與保留的
VRAM。

## 訓練

依照 [config.yaml](config.yaml) 的設定執行完整訓練：

```bash
python train.py
```

快速測試管線可以使用較短的訓練量：

```bash
python train.py --total-timesteps 65536
```

訓練過程會將 TensorBoard event 檔案寫入 `runs/`，並在完成 episode 且
Reward 創新高時，將模型儲存到 `checkpoints/best_model.pt`。

## 查看 TensorBoard

在另一個終端機啟動 TensorBoard：

```bash
tensorboard --logdir runs
```

接著在瀏覽器開啟 `http://localhost:6006`，即可查看：

- `charts/fps`
- `charts/episode_reward`
- `losses/policy_loss`
- `losses/value_loss`
- `system/vram_gb`

## 查看 BipedalWalker 動畫

完成訓練並產生 `checkpoints/best_model.pt` 後，執行：

```bash
python render.py
```

播放 5 個 episode：

```bash
python render.py --episodes 5
```

指定其他 checkpoint 或隨機種子：

```bash
python render.py --checkpoint checkpoints/best_model.pt --seed 123
```

短時間 smoke test 可能尚未完成 episode，因此不一定會產生 checkpoint。
若要觀察學習後的行走效果，請執行完整的 `python train.py`。

## 主要設定

可在 [config.yaml](config.yaml) 修改 GPU、環境數量、rollout 長度、PPO
超參數、TensorBoard 目錄與 checkpoint 目錄。

預設設定如下：

- 32 個平行環境
- 每個環境每次收集 2,048 步
- 每次 rollout 共 65,536 個 transition（`32 * 2048`）
- Minibatch 大小為 2,048
- Learning rate 為 `3e-4`
- Gamma 為 `0.99`

## 專案結構

```text
.
├── benchmark.py       # CUDA 與 VRAM 效能檢查
├── config.yaml        # 硬體、環境與 PPO 設定
├── render.py          # 載入模型並播放動畫
├── requirements.txt   # Python 套件依賴
├── src/
│   ├── agent.py       # GAE 與 PPO 更新邏輯
│   ├── envs.py        # Gymnasium 向量環境封裝
│   └── model.py       # Actor-Critic 網路
└── train.py           # 訓練入口
```

## VRAM 與效能注意事項

Rollout buffer 使用 `float32` tensor。AMP 只會在 CUDA 可用時啟用；如果
`torch.compile()` 因 PyTorch 或執行環境不相容而無法初始化，程式會自動退回
eager mode 繼續訓練。