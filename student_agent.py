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
# import cv2


# class Visualizer:
#     def __init__(self, width=256, height=240, fps=180):
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
        self.feature_extractor = FeatureExtractor(4)
        self.model = DuelingDQN(self.feature_extractor, input_shape=(4, 84, 90), n_actions=len(COMPLEX_MOVEMENT))
        self.model.to(self.device)
        self.model.eval()

        checkpoint = torch.load('mario_dqn_icm.pth1500', map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

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
    
    def act(self, observation):
        self.step_counter += 1

        observation = np.ascontiguousarray(observation)
        processed_frame = self.transform(observation).squeeze(0).numpy()

        # self.visualizer.update(observation)

        if self.first:
            self.frame_stack.clear()
            for _ in range(4):
                self.frame_stack.append(processed_frame)
            self.first = False
        
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