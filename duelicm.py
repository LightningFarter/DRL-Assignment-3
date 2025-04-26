import os
import time
import random
from typing import Tuple, Dict, List
from collections import deque, namedtuple

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import TimeLimit

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

# Hyperparameters
COPY_NETWORK_FREQ: int = 10000
BUFFER_CAPACITY: int = 10000
BATCH_SIZE: int = 32
GAMMA: float = 0.9
LEARNING_RATE: float = 0.00015
ADAM_EPS: float = 0.00015
PER_ALPHA: float = 0.6
PER_BETA_START: float = 0.4
PER_BETA_FRAMES: int = 2000000
PER_EPSILON: float = 0.1
N_STEP: int = 5
NOISY_SIGMA_INIT: float = 2.5
DEATH_PENALTY: int = -100
SKIP_FRAMES: int = 4
MAX_EPISODE_STEPS: int = 3000
MAX_FRAMES: int = 44800000

# ICM Hyperparameters
ICM_EMBED_DIM: int = 256
ICM_BETA: float = 0.1
ICM_ETA: float = 0.01
ICM_LR: float = 1e-4

# Environment Wrappers
class SkipFrame(gym.Wrapper):
    """Skips a specified number of frames, accumulating the reward."""
    def __init__(self, env: gym.Env, skip: int):
        super().__init__(env)
        self._skip = skip

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Steps the environment and accumulates reward over skipped frames."""
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleResize(gym.ObservationWrapper):
    """Converts observations to grayscale and resizes them."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 90)),
            T.ToTensor()
        ])
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1, 84, 90), dtype=np.float32)

    def observation(self, obs: np.ndarray) -> torch.Tensor:
        """Applies the grayscale and resize transformations to the observation."""
        return self.transform(obs)

class FrameStack(gym.Wrapper):
    """Stacks the last k observations along the channel dimension."""
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 1, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Resets the environment and fills the frame stack with the initial observation."""
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return np.concatenate(self.frames, axis=0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Steps the environment and updates the frame stack."""
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(self.frames, axis=0), reward, done, info

def make_env() -> gym.Env:
    """Creates and wraps the Super Mario Bros environment."""
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, SKIP_FRAMES)
    env = GrayScaleResize(env)
    env = FrameStack(env, 4)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return env

# Feature Extractor
class FeatureExtractor(nn.Module):
    """Extracts features from the input image."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the convolutional layers."""
        return self.conv(x)


# Noisy Linear Layer
class NoisyLinear(nn.Module):
    """Linear layer with added Gaussian noise to the weights and biases."""
    def __init__(self, in_f: int, out_f: int, sigma_init: float = NOISY_SIGMA_INIT):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.weight_mu = nn.Parameter(torch.empty(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
        self.register_buffer('weight_epsilon', torch.empty(out_f, in_f))
        self.bias_mu = nn.Parameter(torch.empty(out_f))
        self.bias_sigma = nn.Parameter(torch.empty(out_f))
        self.register_buffer('bias_epsilon', torch.empty(out_f))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initializes the weights and biases."""
        bound = 1 / (self.in_f ** 0.5)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init / (self.in_f ** 0.5))
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_sigma, self.sigma_init / (self.out_f ** 0.5))

    def reset_noise(self):
        """Generates new noise samples for the weights and biases."""
        f = lambda x: x.sign() * x.abs().sqrt()
        eps_in = f(torch.randn(self.in_f))
        eps_out = f(torch.randn(self.out_f))
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass with or without noise depending on training mode."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network architecture."""
    def __init__(self, feature_extractor: nn.Module, input_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.n_actions = n_actions
        with torch.no_grad():
            # Move the dummy tensor to the same device as the feature extractor
            dummy = torch.zeros(1, *input_shape, device=next(feature_extractor.parameters()).device)
            feat_dim = self.feature_extractor(dummy).view(1, -1).size(1)
        self.value = nn.Sequential(
            NoisyLinear(feat_dim, 512), nn.ReLU(),
            NoisyLinear(512, 1)
        )
        self.advantage = nn.Sequential(
            NoisyLinear(feat_dim, 512), nn.ReLU(),
            NoisyLinear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass, returning Q-values."""
        x = self.feature_extractor(x / 255.0).view(x.size(0), -1)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self):
        """Resets the noise in the value and advantage streams."""
        for module in self.value:
            if hasattr(module, 'reset_noise'):
                module.reset_noise()
        for module in self.advantage:
            if hasattr(module, 'reset_noise'):
                module.reset_noise()


# Intrinsic Curiosity Module
class ICM(nn.Module):
    """Intrinsic Curiosity Module to encourage exploration."""
    def __init__(self, feature_extractor: nn.Module, input_shape: Tuple[int, int, int], n_actions: int, embed_dim: int = ICM_EMBED_DIM):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.n_actions = n_actions
        with torch.no_grad():
            # Move the dummy tensor to the same device as the feature extractor
            dummy = torch.zeros(1, *input_shape, device=next(feature_extractor.parameters()).device)
            feat_dim = self.feature_extractor(dummy).view(1, -1).size(1)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU()
        )
        self.inverse = nn.Sequential(
            nn.Linear(embed_dim * 2, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(embed_dim + n_actions, 512), nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the ICM, predicting action and next state embedding."""
        feat = self.feature_extractor(state).detach()
        next_feat = self.feature_extractor(next_state).detach()
        phi = self.encoder(feat)
        phi_next = self.encoder(next_feat)
        inv_in = torch.cat([phi, phi_next], dim=1)
        logits = self.inverse(inv_in)
        a_onehot = F.one_hot(action, self.n_actions).float()
        fwd_in = torch.cat([phi, a_onehot], dim=1)
        pred_phi_next = self.forward_model(fwd_in)
        inv_loss = F.cross_entropy(logits, action)
        fwd_loss = F.mse_loss(pred_phi_next, phi_next)
        return inv_loss, fwd_loss, pred_phi_next, phi_next

    def intrinsic_reward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Calculates the intrinsic reward based on the prediction error of the forward model."""
        with torch.no_grad():
            feat = self.feature_extractor(state).detach()
            next_feat = self.feature_extractor(next_state).detach()
            phi = self.encoder(feat)
            phi_next = self.encoder(next_feat)
            a_onehot = F.one_hot(action, self.n_actions).float()
            fwd_in = torch.cat([phi, a_onehot], dim=1)
            pred_phi_next = self.forward_model(fwd_in)
            return 0.5 * (pred_phi_next - phi_next).pow(2).sum(dim=1)

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    """Replay buffer with prioritized experience replay."""
    def __init__(self, cap: int, alpha: float, beta_start: float, beta_frames: int, n_step: int, gamma: float):
        self.cap = cap
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta_by_frame = lambda f: min(1.0, beta_start + f * (1.0 - beta_start) / beta_frames)
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: List = []
        self.prios = np.zeros((cap,), dtype=np.float32)
        self.pos = 0
        self.n_buf = deque(maxlen=n_step)
        self.Exp = namedtuple('Exp', ['s', 'a', 'r', 's2', 'd'])

    def _get_n_step(self) -> Tuple[float, np.ndarray, bool]:
        """Calculates the n-step return and next state."""
        reward, next_state, done = self.n_buf[-1].r, self.n_buf[-1].s2, self.n_buf[-1].d
        for transition in reversed(list(self.n_buf)[:-1]):
            reward = transition.r + self.gamma * reward * (1 - transition.d)
            next_state, done = (transition.s2, transition.d) if transition.d else (next_state, done)
        return reward, next_state, done

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: bool):
        """Adds a transition to the n-step buffer."""
        self.n_buf.append(self.Exp(s, a, r, s2, d))
        if len(self.n_buf) < self.n_step:
            return
        reward_n, next_state_n, done_n = self._get_n_step()
        state_0, action_0 = self.n_buf[0].s, self.n_buf[0].a
        experience = self.Exp(state_0, action_0, reward_n, next_state_n, done_n)
        if len(self.buffer) < self.cap:
            self.buffer.append(experience)
            priority = 1.0 if len(self.buffer) == 1 else self.prios.max()
        else:
            self.buffer[self.pos] = experience
            priority = self.prios.max()
        self.prios[self.pos] = priority
        self.pos = (self.pos + 1) % self.cap
        self.n_buf.popleft()

    def sample(self, batch_size: int, frame_idx: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples a batch of experiences from the buffer with priorities."""
        N = len(self.buffer)
        if N == 0:
            return [], [], [], [], [], [], []
        priorities = self.prios[:N] ** self.alpha
        sum_p = priorities.sum()
        probabilities = priorities / sum_p if sum_p > 0 else np.ones_like(priorities) / N
        indices = np.random.choice(N, batch_size, p=probabilities)
        batch = self.Exp(*zip(*[self.buffer[i] for i in indices]))
        beta = self.beta_by_frame(frame_idx)
        weights = (N * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return (np.array(batch.s), np.array(batch.a), np.array(batch.r, dtype=np.float32),
                np.array(batch.s2), np.array(batch.d, dtype=np.float32),
                weights.astype(np.float32), np.array(indices))

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """Updates the priorities of the sampled experiences."""
        for i, error in zip(indices, errors):
            self.prios[i] = abs(error) + PER_EPSILON

# Agent
class Agent:
    """The reinforcement learning agent."""
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int, device: torch.device):
        self.device = device
        self.n_actions = n_actions
        self.feature_extractor = FeatureExtractor(obs_shape[0]).to(device)
        self.online = DuelingDQN(self.feature_extractor, obs_shape, n_actions).to(device)
        self.target_feature_extractor = FeatureExtractor(obs_shape[0]).to(device)
        self.target = DuelingDQN(self.target_feature_extractor, obs_shape, n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)
        self.icm = ICM(self.feature_extractor, obs_shape, n_actions).to(device)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=ICM_LR)
        self.buffer = PrioritizedReplayBuffer(BUFFER_CAPACITY, PER_ALPHA, PER_BETA_START,
                                             PER_BETA_FRAMES, N_STEP, GAMMA)
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.frame_idx = 0
        self.update_freq = COPY_NETWORK_FREQ

    def act(self, state: np.ndarray) -> int:
        """Selects an action based on the current state."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state_tensor)
        return int(q_values.argmax(1).item())

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Adds a transition to the replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self):
        """Performs a learning step to update the online network."""
        if len(self.buffer.buffer) < self.batch_size:
            return

        states, actions, rewards_ext, next_states, dones, weights, indices = self.buffer.sample(
            self.batch_size, self.frame_idx, self.device
        )

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        rewards_ext_tensor = torch.tensor(rewards_ext, dtype=torch.float32, device=self.device)

        # Calculate intrinsic reward and ICM loss
        inv_loss, fwd_loss, predicted_phi_next, target_phi_next = self.icm(
            states_tensor, next_states_tensor, actions_tensor
        )
        intrinsic_reward = ICM_ETA * 0.5 * (predicted_phi_next - target_phi_next).pow(2).sum(dim=1)
        icm_loss = (1 - ICM_BETA) * inv_loss + ICM_BETA * fwd_loss

        # Calculate target Q-values
        q_predicted = self.online(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_actions = self.online(next_states_tensor).argmax(1)
        q_next = self.target(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        total_reward = rewards_ext_tensor + intrinsic_reward
        q_target = total_reward + (self.gamma ** N_STEP) * q_next * (1 - dones_tensor)

        # Calculate DQN loss
        td_error = q_predicted - q_target.detach()
        dqn_loss = (F.smooth_l1_loss(q_predicted, q_target.detach(), reduction='none') * weights_tensor).mean()

        # Update online network
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        self.online.reset_noise()
        self.target.reset_noise()

        # Update ICM network
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

        # Update priorities in the buffer
        self.buffer.update_priorities(indices, td_error.detach().cpu().numpy())

        # Update target network
        if self.frame_idx % self.update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

# Training Loop
def train(num_episodes: int, checkpoint_path: str = 'checkpoints/mario_dqn_icm.pth') -> Dict[str, List]:
    """Trains the agent in the Super Mario Bros environment."""
    env = make_env()
    print(f"Observation space shape: {env.observation_space.shape}")
    agent = Agent(env.observation_space.shape, env.action_space.n,
                  torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    start_episode, initial_frame = 1, 0

    # Load checkpoint if it exists
    if os.path.isfile(checkpoint_path + "1200"):
        checkpoint = torch.load(checkpoint_path + "1200", map_location=agent.device)
        agent.online.load_state_dict(checkpoint['model'])
        agent.target.load_state_dict(checkpoint['model'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.icm_optimizer.load_state_dict(checkpoint['icm_opt'])
        initial_frame = checkpoint.get('frame_idx', 0)
        start_episode = checkpoint.get('episode', 0) + 1
        print(f"Loaded checkpoint from episode {start_episode - 1} at frame {initial_frame}")

    agent.frame_idx = initial_frame
    raw_state = env.reset()
    current_state = raw_state

    # Fill the buffer with initial experiences
    print("Filling replay buffer...")
    while len(agent.buffer.buffer) < BATCH_SIZE:
        action = agent.act(current_state)
        next_state, reward, done, _ = env.step(action)
        agent.push(current_state, action, reward, next_state, done)
        current_state = next_state
        if done:
            current_state = env.reset()
    print("Replay buffer filled.")

    history = {'reward': [], 'env_reward': [], 'stage': [], 'Trun': []}
    start_time = time.time()

    for episode in range(start_episode, num_episodes + 1):
        observation = env.reset()
        # Perform some initial random actions to introduce variance
        for _ in range(8):
            observation, _, _, _ = env.step(env.action_space.sample())
        current_state = observation
        episode_reward, episode_env_reward, previous_life = 0, 0, None
        done = False

        while not done:
            agent.frame_idx += 1
            action = agent.act(current_state)
            next_state, env_reward, done, info = env.step(action)
            truncated = info.get('TimeLimit.truncated', False)
            done_flag = done and not truncated
            current_reward = env_reward
            current_life = info.get('life')

            if previous_life is None:
                previous_life = current_life
            elif current_life < previous_life:
                current_reward += DEATH_PENALTY
                done = True
            previous_life = current_life

            agent.push(current_state, action, current_reward, next_state, done_flag)
            agent.learn()
            current_state = next_state
            episode_reward += current_reward
            episode_env_reward += env_reward

        termination_status = "TERMINATED" if done else "TRUNCATED"
        history['reward'].append(episode_reward)
        history['env_reward'].append(episode_env_reward)
        history['stage'].append(env.unwrapped._stage)
        history['Trun'].append(termination_status)

        print(f"Episode: {episode} | Env Reward: {episode_env_reward:.2f} | Custom Reward: {episode_reward:.2f} | Stage: {env.unwrapped._stage}")

        if episode % 100 == 0:
            duration = time.time() - start_time
            start_time = time.time()
            avg_env_reward = np.mean(history['env_reward'][-100:])
            avg_custom_reward = np.mean(history['reward'][-100:])
            avg_stage = np.mean(history['stage'][-100:])
            truncated_episodes = history['Trun'][-100:].count('TRUNCATED')
            print(f"[Ep {episode}] | Avg EnvR: {avg_env_reward:.2f} | Avg CustR: {avg_custom_reward:.2f} | Avg Stage: {avg_stage:.2f} | Truncated: {truncated_episodes} | Time: {duration / 60:.2f} min")
            torch.save({'model': agent.online.state_dict(), 'optimizer': agent.optimizer.state_dict(),
                        'icm_opt': agent.icm_optimizer.state_dict(), 'frame_idx': agent.frame_idx, 'episode': episode},
                       checkpoint_path + f'{episode}')

            # Plotting rewards
            x_axis = [i * 100 for i in range(1, len(history['env_reward']) // 100 + 1)]
            avg_env_rewards = [np.mean(history['env_reward'][(i - 1) * 100:i * 100]) for i in range(1, len(x_axis) + 1)]
            avg_custom_rewards = [np.mean(history['reward'][(i - 1) * 100:i * 100]) for i in range(1, len(x_axis) + 1)]
            plt.figure(figsize=(10, 5))
            plt.plot(x_axis, avg_env_rewards, marker='o', label='Avg Env Reward')
            plt.plot(x_axis, avg_custom_rewards, marker='x', label='Avg Custom Reward')
            plt.xlabel('Episodes')
            plt.ylabel('Average Reward (per 100 episodes)')
            plt.title('Average Reward Over Episodes')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(os.path.dirname(checkpoint_path), 'average_reward.png'))
            plt.close()

    print("Training complete.")
    return history

if __name__ == '__main__':
    train(num_episodes=100000)