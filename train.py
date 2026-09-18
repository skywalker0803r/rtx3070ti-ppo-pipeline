"""Train PPO on a vectorized BipedalWalker environment."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.agent import PPOAgent, Rollout, compute_gae
from src.envs import clip_actions, create_vector_env
from src.model import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    hardware = config["hardware"]
    environment = config["environment"]
    hyper = config["hyperparameters"]
    seed = hyper["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if hardware["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    num_envs = environment["num_envs"]
    num_steps = hyper["num_steps"]
    total_timesteps = args.total_timesteps or hyper["total_timesteps"]
    envs = create_vector_env(environment["env_id"], num_envs, seed)
    observations, _ = envs.reset(seed=seed)
    observation_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    model = ActorCritic(observation_dim, action_dim).to(device)
    if hardware["compile_model"] and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as error:
            print(f"torch.compile unavailable, continuing eagerly: {error}")
    agent = PPOAgent(model, hyper["learning_rate"], device, hardware["enable_amp"])
    log_dir = Path(config["monitoring"]["log_dir"])
    checkpoint_dir = Path(config["monitoring"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    best_reward = -float("inf")
    global_step = 0
    start_time = time.perf_counter()
    updates = max(1, total_timesteps // (num_envs * num_steps))
    for update in trange(updates, desc="PPO updates"):
        obs_buffer = torch.empty((num_steps, num_envs, observation_dim), dtype=torch.float32, device=device)
        action_buffer = torch.empty((num_steps, num_envs, action_dim), dtype=torch.float32, device=device)
        logprob_buffer = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        reward_buffer = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        done_buffer = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        value_buffer = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        episode_rewards = []
        for step in range(num_steps):
            observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions, logprobs, _, values = model.get_action_and_value(observation_tensor)
            action_numpy = clip_actions(actions.float().cpu().numpy(), envs.single_action_space)
            next_observations, rewards, terminated, truncated, info = envs.step(action_numpy)
            dones = np.logical_or(terminated, truncated)
            obs_buffer[step] = observation_tensor
            action_buffer[step] = actions
            logprob_buffer[step] = logprobs
            reward_buffer[step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            done_buffer[step] = torch.as_tensor(dones, dtype=torch.float32, device=device)
            value_buffer[step] = values
            observations = next_observations
            global_step += num_envs
            if "final_info" in info:
                for final_info in info["final_info"]:
                    if final_info and "episode" in final_info:
                        episode_rewards.append(final_info["episode"]["r"])
        with torch.no_grad():
            next_value = model.get_value(torch.as_tensor(observations, dtype=torch.float32, device=device))
        advantages, returns = compute_gae(reward_buffer, done_buffer, value_buffer, next_value, environment["gamma"], environment["gae_lambda"])
        metrics = agent.update(Rollout(obs_buffer, action_buffer, logprob_buffer, reward_buffer, done_buffer, value_buffer), advantages, returns, hyper["batch_size"], hyper["n_epochs"], hyper["clip_coef"], hyper["ent_coef"], hyper["vf_coef"], hyper["max_grad_norm"])
        elapsed = time.perf_counter() - start_time
        fps = global_step / max(elapsed, 1e-6)
        writer.add_scalar("charts/fps", fps, global_step)
        writer.add_scalar("losses/policy_loss", metrics["policy_loss"], global_step)
        writer.add_scalar("losses/value_loss", metrics["value_loss"], global_step)
        if torch.cuda.is_available():
            writer.add_scalar("system/vram_gb", torch.cuda.memory_allocated() / 1024**3, global_step)
        if episode_rewards:
            mean_reward = float(np.mean(episode_rewards))
            writer.add_scalar("charts/episode_reward", mean_reward, global_step)
            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save({"model": model.state_dict(), "config": config, "reward": best_reward}, checkpoint_dir / "best_model.pt")
    writer.close()
    envs.close()
    print(f"Completed {global_step:,} steps at {global_step / max(time.perf_counter() - start_time, 1e-6):,.0f} FPS on {device}.")


if __name__ == "__main__":
    main()