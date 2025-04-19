import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import pygame
from pygame.locals import *
import pickle

# Environment preprocessing wrapper
class PreprocessEnv(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, n_frames=4):
        super(PreprocessEnv, self).__init__(env)
        self.width = width
        self.height = height
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(n_frames, height, width),
            dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        obs = self._process_obs(obs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0)

    def observation(self, obs):
        obs = self._process_obs(obs)
        self.frames.append(obs)
        return np.stack(self.frames, axis=0)

    def _process_obs(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return obs

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._current_score = 0
        self._current_x_pos = 0
        self._current_lives = 3

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        # Custom rewards
        reward = 0.0
        
        # Reward for moving right
        if info["x_pos"] > self._current_x_pos:
            reward += (info["x_pos"] - self._current_x_pos) * 0.1
        
        # Penalty for dying
        if info["life"] < self._current_lives:
            reward -= 500
        
        # Reward for collecting coins
        if info["score"] > self._current_score:
            reward += (info["score"] - self._current_score) * 0.01
        
        # Update stored values
        self._current_x_pos = info["x_pos"]
        self._current_score = info["score"]
        self._current_lives = info["life"]
        
        return state, reward, done, info

# Experience replay buffers
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
        else:
            print(f"Warning: No existing replay buffer at {path}")

# Dueling DQN architecture
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        advantage = self.advantage(conv_out)
        value = self.value(conv_out)
        return value + advantage - advantage.mean(1, keepdim=True)

# DQN Agent
class Agent(object):
    def __init__(self, input_shape, num_actions, use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model = DQN(input_shape, num_actions).to(self.device)
        self.target_model = DQN(input_shape, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025, eps=1e-4)
        self.action_space = gym.spaces.Discrete(num_actions)
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.batch_size = 128
        self.gamma = 0.99
        self.target_update = 10000
        self.steps = 0

        self.load_checkpoint('mario_1300.pth')

    def act(self, observation):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        obs = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(obs)
        return q_values.argmax().item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, agent_buffer, expert_buffer):
        if len(agent_buffer) < self.batch_size//2 or len(expert_buffer) < 1:
            return 0.0

        # Sample from both buffers
        agent_samples = min(self.batch_size//2, len(agent_buffer))
        expert_samples = self.batch_size - agent_samples
        a_states, a_actions, a_rewards, a_next_states, a_dones = agent_buffer.sample(agent_samples)
        e_states, e_actions, e_rewards, e_next_states, e_dones = expert_buffer.sample(expert_samples)

        # Combine and convert to tensors
        states = torch.FloatTensor(np.array(a_states + e_states)).to(self.device)
        actions = torch.LongTensor(a_actions + e_actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(a_rewards + e_rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(a_next_states + e_next_states)).to(self.device)
        dones = torch.FloatTensor(a_dones + e_dones).unsqueeze(1).to(self.device)

        # Compute Q-values and target values with Double DQN
        current_q = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Calculate loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target network
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.steps += 1
        self.update_epsilon()

        return loss.item()

    def save_checkpoint(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'target_state': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.target_model.load_state_dict(checkpoint['target_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

# Human expert demonstration recorder
def record_human_expert(
    expert_buffer,
    num_episodes=1,
    save_path="expert_buffer.pkl",
    append_mode=True
):
    if not append_mode and os.path.exists(save_path):
        expert_buffer.load(save_path)
        print("Loaded existing expert buffer.")
        return
    
    expert_buffer.load(save_path)

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = PreprocessEnv(env)
    
    pygame.init()
    screen = pygame.display.set_mode((256, 240))
    pygame.display.set_caption("Human Expert")
    clock = pygame.time.Clock()
    
    episode = 0
    while episode < num_episodes:
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = 0
            keys = pygame.key.get_pressed()
            
            # Key to action mapping
            pressed = []
            if keys[K_RIGHT]: pressed.append('right')
            if keys[K_LEFT]: pressed.append('left')
            if keys[K_UP]: pressed.append('up')
            if keys[K_DOWN]: pressed.append('down')
            if keys[K_a]: pressed.append('A')
            if keys[K_s]: pressed.append('B')
            
            # Find matching action index
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
            
            for i, buttons in enumerate(COMPLEX_MOVEMENT):
                if sorted(pressed) == sorted(buttons):
                    action = i
                    break
            
            next_state, reward, done, info = env.step(action)
            expert_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Render
            img = env.render(mode='rgb_array')
            img = np.transpose(img, (1, 0, 2))
            surf = pygame.surfarray.make_surface(img)
            screen.blit(surf, (0, 0))
            pygame.display.update()
            clock.tick(30)
        
        episode += 1
        print(f"Expert Episode {episode} - Reward: {total_reward}")
    
    pygame.quit()
    env.close()
    expert_buffer.save(save_path)  # Save after recording
    print(f"Saved expert buffer to {save_path}")

# Training process
def train(
    agent,
    env,
    agent_buffer,
    expert_buffer,
    expert_mode="append",  # "append" or "replace"
    expert_path="expert_buffer.pkl",
    episodes=10000,
    save_interval=100
):
    # Load existing expert data if available
    if os.path.exists(expert_path):
        if expert_mode == "replace":
            expert_buffer.load(expert_path)
            print("Using existing expert buffer (replacing current).")
        elif expert_mode == "append":
            temp_buffer = ReplayBuffer(expert_buffer.capacity)
            temp_buffer.load(expert_path)
            for sample in temp_buffer.buffer:
                expert_buffer.push(*sample)
            print(f"Appended {len(temp_buffer)} expert samples.")
    
    # Rest of training loop remains the same...
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent_buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            
            loss = agent.train(agent_buffer, expert_buffer)
        
        if ep % save_interval == 0:
            path = os.path.join(checkpoint_dir, f"mario_{ep}.pth")
            agent.save_checkpoint(path)
            print(f"Saved checkpoint at episode {ep}")
        
        print(f"Episode {ep} - Reward: {total_reward} - Epsilon: {agent.epsilon:.4f}")

if __name__ == "__main__":
    # Initialize environment and agent
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = PreprocessEnv(env)
    
    # Initialize buffers
    agent_buffer = ReplayBuffer(100000)
    expert_buffer = ReplayBuffer(100000)

    expert_buffer.load("exprt_buffer.pkl")
    
    # Record human demonstrations
    print("Recording human expert...")
    record_human_expert(expert_buffer, num_episodes=1, append_mode=True)
    print(len(expert_buffer.buffer))
    
    # Create agent
    agent = Agent(env.observation_space.shape, env.action_space.n)
    agent.load_checkpoint("checkpoints/mario_700.pth")
    
    # Start training
    print("Starting training...")
    train(agent, env, agent_buffer, expert_buffer, expert_mode="append")