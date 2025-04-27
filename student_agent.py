import os
import time
import random
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
from torchvision import transforms as T

from duelicm import FeatureExtractor, DuelingDQN

# import pygame
import pickle
from sklearn.metrics import mean_squared_error


# class Visualizer:
#     def __init__(self, width=256, height=240, fps=300):
#         pygame.init()
#         self.width = width
#         self.height = height
#         self.fps = fps
#         self.screen = pygame.display.set_mode((width, height))
#         pygame.display.set_caption("Agent Visualization")
#         self.clock = pygame.time.Clock()

#     def update(self, frame):
#         """
#         Update the Pygame display with the given frame.

#         Args:
#             frame (np.ndarray): The frame to display, expected in HWC format (Height, Width, Channels).
#         """
#         if frame.shape[0] == 3:  # CHW format
#             frame = np.transpose(frame, (1, 2, 0))  # Convert to HWC format
        
#         frame = np.transpose(frame, (1, 0, 2))

#         # Convert the frame to a Pygame surface
#         surface = pygame.surfarray.make_surface(frame)
#         surface = pygame.transform.scale(surface, (self.width, self.height))

#         # Display the surface
#         self.screen.blit(surface, (0, 0))
#         pygame.display.flip()
#         self.clock.tick(self.fps)

#     def close(self):
#         """Close the Pygame window."""
#         pygame.quit()


class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor1 = FeatureExtractor(4)
        self.model1 = DuelingDQN(self.feature_extractor1, input_shape=(4, 84, 90), n_actions=len(COMPLEX_MOVEMENT))
        self.model1.to(self.device)
        self.feature_extractor2 = FeatureExtractor(4)
        self.model2 = DuelingDQN(self.feature_extractor2, input_shape=(4, 84, 90), n_actions=len(COMPLEX_MOVEMENT))
        self.model1.eval()
        self.model2.eval()


        # checkpoint = torch.load('pulltest.pth', map_location=self.device)
        checkpoint = torch.load('mario_dqn_icm1500.pth', map_location=self.device)
        self.model1.load_state_dict(checkpoint['model'])
        self.model1.to(self.device)
        checkpoint = torch.load('mario_dqn_icm_lvl2.pth3700', map_location=self.device)
        self.model2.load_state_dict(checkpoint['model'])
        self.model2.to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 90)),
            T.ToTensor()
        ])

        self.frame_stack = deque(maxlen=4)
        self.first = True

        self.skip_frames = 3
        self.skip_count = 0
        self.last_action = 0
        self.step_counter = 0

        # self.visualizer = Visualizer()
        
        with open('lvl1_init.pkl', 'rb') as f:
            self.lvl1_init = pickle.load(f)
        with open('lvl2_init.pkl', 'rb') as f:
            self.lvl2_init = pickle.load(f)
        self.lvl = 1

        self.model = self.model1
        self.change_model = True
        self.wait = -1

        self.reset_wait = 1000
    
    def act(self, observation):
        self.step_counter += 1
        self.reset_wait -= 1

        observation = np.ascontiguousarray(observation)
        processed_frame = self.transform(observation).squeeze(0).numpy()

        # self.visualizer.update(observation)

        if self.reset_wait <= 0:
            mse_lvl1 = mean_squared_error(observation.flatten(), self.lvl1_init.flatten())
            if mse_lvl1 < 0.5:
                self.__init__()

        if self.wait > 0:
            self.wait -= 1
            return self.last_action

        if self.change_model:
            mse_lvl1 = mean_squared_error(observation.flatten(), self.lvl1_init.flatten())
            mse_lvl2 = mean_squared_error(observation.flatten(), self.lvl2_init.flatten())

            if mse_lvl2 < mse_lvl1 and mse_lvl2 < 0.3:
                self.change_model = False
                self.model = self.model2
                self.wait = 10
                self.frame_to_1 = 1100
            else:
                self.model = self.model1

        if self.first:
            self.frame_stack.clear()
            for _ in range(4):
                self.frame_stack.append(processed_frame)
            self.first = False
            self.reset_wait = 1000
        
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        self.frame_stack.append(processed_frame)
        stacked_frames = np.stack(self.frame_stack, axis=0)
        obs_tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(obs_tensor)
            action = q_values.argmax(dim=1).item()

        self.last_action = action
        self.skip_count = self.skip_frames

        return action

    def reset(self):
        self.first = True
        self.skip_count = 0
        self.last_action = 0
        self.step_counter = 0


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    agent = Agent()
    for i in range(3):
        done = False
        total_reward = 0
        state = env.reset()
        steps = 0
        while not done:
            steps += 1
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        print(f"total reward: {total_reward}, steps: {steps}")


if __name__ == "__main__":
    main()