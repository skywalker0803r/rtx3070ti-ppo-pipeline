# RTX 3070 Ti PPO Pipeline

A lightweight PyTorch PPO implementation for training on Gymnasium's
`BipedalWalker-v3`, tuned for an 8 GB NVIDIA RTX 3070 Ti.

## Features

- 32-process `AsyncVectorEnv` rollout collection
- Orthogonally initialized actor-critic network
- GAE advantages and clipped PPO updates
- CUDA, AMP, and optional `torch.compile()` support
- TensorBoard metrics for FPS, losses, rewards, and VRAM
- Best-model checkpointing to `checkpoints/best_model.pt`

## Requirements

- Linux with Python 3.10+
- NVIDIA driver with CUDA support
- NVIDIA RTX 3070 Ti or another CUDA-capable GPU

The project dependencies are listed in [requirements.txt](requirements.txt).

## Installation

On Debian or Ubuntu, install virtual-environment support if needed:

```bash
sudo apt install python3-venv
```

Create and activate the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Hardware Benchmark

Verify that PyTorch detects the RTX 3070 Ti before training:

```bash
python benchmark.py
```

The benchmark reports the GPU name, total VRAM, matrix-multiplication throughput,
and allocated/reserved VRAM.

## Training

Run the configured training job:

```bash
python train.py
```

For a short smoke test:

```bash
python train.py --total-timesteps 65536
```

Training writes TensorBoard event files under `runs/` and saves a new best model
under `checkpoints/` whenever an episode reward improves.

Launch TensorBoard in a separate terminal:

```bash
tensorboard --logdir runs
```

## Configuration

Edit [config.yaml](config.yaml) to change the device, environment count,
rollout length, PPO hyperparameters, logging directory, or checkpoint directory.
The default rollout contains 65,536 transitions (`32 * 2048`) and uses a 2,048
sample minibatch to balance GPU utilization and the 8 GB VRAM limit.

## Project Layout

```text
.
├── benchmark.py       # CUDA and VRAM benchmark
├── config.yaml        # Hardware, environment, and PPO settings
├── requirements.txt   # Python dependencies
├── src/
│   ├── agent.py       # GAE and PPO update logic
│   ├── envs.py        # Vectorized Gymnasium environments
│   └── model.py       # Actor-critic network
└── train.py           # Training entry point
```

## Notes

Rollout buffers use `float32` tensors and are allocated on the selected device.
AMP is enabled automatically only when CUDA is available. If `torch.compile()`
cannot be initialized on the installed PyTorch/runtime combination, training
continues in eager mode.