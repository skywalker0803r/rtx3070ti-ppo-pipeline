"""Render a trained PPO policy in the BipedalWalker window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import gymnasium as gym

from src.model import ActorCritic


def load_model(checkpoint_path: Path, observation_dim: int, action_dim: int, device: torch.device) -> ActorCritic:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ActorCritic(observation_dim, action_dim).to(device)
    state_dict = checkpoint["model"]
    state_dict = {
        key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    reward = checkpoint.get("reward")
    if reward is not None:
        print(f"Loaded checkpoint reward: {float(reward):.2f}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trained BipedalWalker PPO policy.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.0, help="Optional delay between frames in seconds")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}. Run a longer training job first with `python train.py`."
        )
    with open(args.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    device = torch.device("cuda" if config["hardware"]["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    env = gym.make(config["environment"]["env_id"], render_mode="human")
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    model = load_model(checkpoint_path, observation_dim, action_dim, device)
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = model.actor(observation_tensor).squeeze(0).cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if args.delay > 0:
                    time.sleep(args.delay)
            print(f"Episode {episode + 1}: reward={total_reward:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()