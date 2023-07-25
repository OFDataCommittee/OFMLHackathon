from environment import RBCEnv

import gymnasium as gym
from gymnasium.vector.async_vector_env import AsyncVectorEnv

from gymprecice.utils.constants import EPSILON, LOG_EPSILON
from gymprecice.utils.multienvutils import worker_with_lock
from gymprecice.utils.fileutils import make_result_dir

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal

import numpy as np
import warnings
import math
import time
import logging
import random

from typing import Optional
import argparse
from distutils.util import strtobool

import time

try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_actions = np.prod(env.single_action_space.shape)
        self.n_obs = np.prod(env.single_observation_space.shape)

        self.action_min = torch.from_numpy(np.copy(env.single_action_space.low))
        self.action_max = torch.from_numpy(np.copy(env.single_action_space.high))

        self.action_scale = (self.action_max - self.action_min) / 2.0
        self.action_bias = (self.action_max + self.action_min) / 2.0

        self.latent_dim = 64

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.n_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.latent_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.latent_dim, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.n_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.latent_dim)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(
            nn.Linear(self.latent_dim, self.n_actions), std=0.1
        )
        self.std = nn.Parameter(torch.zeros(self.n_actions), requires_grad=True)

    def get_value(self, x):
        x = x.reshape(-1, self.n_obs)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.reshape(-1, self.n_obs)
        latent_pi = self.actor(x)
        mean = self.actor_mean(latent_pi)

        std = self.std.expand_as(mean)
        # Clip stddev for numerical stability (epsilon < 1.0, hence negative)
        std = torch.clip(std, LOG_EPSILON, -LOG_EPSILON)
        # Softplus transformation (based on https://arxiv.org/abs/2007.06059)
        std = 0.25 * (torch.log(1.0 + torch.exp(std)) + 0.2) / (math.log(2.0) + 0.2)
        probs = Normal(mean, std)
        sample = probs.rsample()

        if action is None:
            squashed_sample = torch.tanh(sample)
            action = (
                squashed_sample * self.action_scale + self.action_bias
            )  # we scale the sampled action

        # TODO: this should be done only when action is not None
        squashed_action = (
            2.0 * (action - self.action_min) / (self.action_max - self.action_min) - 1.0
        )
        clip = 1.0 - EPSILON
        squashed_action = torch.clip(squashed_action, -clip, clip)

        gaussian_action = torch.atanh(squashed_action)

        log_prob = probs.log_prob(gaussian_action)
        log_prob -= 2.0 * (
            math.log(2.0)
            - gaussian_action
            - torch.log(1.0 + torch.exp(-2.0 * gaussian_action))
        )

        entropy = probs.entropy().sum(1)
        # agent returns the mean action to be used for deterministic evaluation
        return action, mean, log_prob.sum(1), entropy, self.critic(x)


class WandBRewardRecoder(gym.Wrapper):
    """This wrapper will keep track of cumulative rewards and episode lengths"""

    def __init__(self, env: gym.Env, wandb_context=None):
        """This wrapper will keep track of cumulative rewards and episode lengths.
        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        super().__init__(env)
        self.episode_count = 0
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.wandb_context = wandb_context

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        observations, infos = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, infos

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        observations, rewards, dones, _, infos = self.env.step(action)
        self.episode_returns += rewards.flatten()
        self.episode_lengths += 1

        if self.num_envs == 1:
            dones = [dones]
        dones = list(dones)

        for i in range(len(dones)):
            if dones[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                if self.wandb_context:
                    metrics_dict = {
                        "rewards": episode_return / episode_length,
                        "episode": self.episode_count,
                    }
                    self.wandb_context.log(metrics_dict, commit=True)
                print(
                    f"episode: {self.episode_count}, rewards : {episode_return / episode_length}"
                )
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        terminated = (dones if self.num_envs > 1 else dones[0],)
        truncated = False
        return (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        )

    def close(self):
        self.env.close()


def parse_args():
    # Training specific arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=12345, help="seed of the experiment"
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--dump-policy",
        type=bool,
        default=True,
        help="tif toggled, trained policy will be dumped each `dump_policy_freq`",
    )
    parser.add_argument(
        "--dump-policy-freq",
        type=int,
        default=10,
        help="the freqency of saving policy on hard-drive",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="RL_CFD",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="cfddrl",
        help="the entity (team) of wandb's project",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--env-id", type=str, default="", help="the id of the environment"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1440000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Use GAE for advantage computation",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=2, help="the number of mini-batches"
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=10,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=1e-2, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="coefficient of the value function"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    args.track = False

    return args



if __name__ == "__main__":
    start_time = time.time()

    environment_config = make_result_dir()
    args = parse_args()
    # weigh and biases
    wandb_recorder = None
    if args.track:
        try:
            import wandb
            run_name = f'{environment_config["environment"].get("name", "training")}_{int(time.time())}'
            wandb_recorder = wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=vars(args),
                name=run_name,
                monitor_gym=False,
                save_code=True,
            )
        except ImportError as err:
            logger.error("wandb is not installed, run `pip install gymprecice[vis]`")
            raise err

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    def make_env(options, idx, wrappers=None):
        def _make_env():
            env = RBCEnv(options, idx)
            if wrappers is not None:
                if callable(wrappers):
                    env = wrappers(env)
                elif isinstance(wrappers, Iterable) and all(
                    [callable(w) for w in wrappers]
                ):
                    for wrapper in wrappers:
                        env = wrapper(env)
                else:
                    raise NotImplementedError
            return env
        return _make_env

    env_fns = []
    for idx in range(args.num_envs):
        env_fns.append(
            make_env(
                options=environment_config, idx=idx, wrappers=[gym.wrappers.ClipAction]
            )
        )

    # env setup
    envs = AsyncVectorEnv(
        env_fns=env_fns, context="fork", shared_memory=False, worker=worker_with_lock
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    envs = WandBRewardRecoder(envs, wandb_recorder)

    obs_dim = np.prod(envs.single_observation_space.shape)
    n_acts = np.prod(envs.single_action_space.shape)
    device = "cpu"
    agent = Agent(envs)
    optimizer = Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    # ALGO Logic: Storage setup --> (timesteps, num_env, n_obs) if obs is 2d then the following will be 4 dimesional
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    nxtobs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)

    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # reset returns info as well
    next_obs = torch.Tensor(envs.reset()[0]).to(device)

    for update in range(1, args.num_updates + 1):
        print(f"Policy update#{update} in progress ...")
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, _, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, _, info = envs.step(action)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(np.array(done)).to(device=device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = -entropy.mean()

                loss = pg_loss + args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    print(f"approx_kl is violated break at update_epochs {epoch}")
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        metrics_dict = {
            "update": update,
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "global_step": global_step,
            "charts/SPS": int(global_step / (time.time() - start_time)),
        }

        if wandb_recorder is not None:
            wandb_recorder.log(metrics_dict, commit=True)

        if args.dump_policy and update % args.dump_policy_freq == 0:
            torch.save(agent.state_dict(), f"policy_{update}.pt")
        
        print(f"Policy update#{update} completed.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time/3600} hours")
    print("End of training!")

    envs.close()