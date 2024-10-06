import gym
from gym import spaces
import numpy as np
import pygame

class SpaceShooterEnv(gym.Env):
    def __init__(self):
        super(SpaceShooterEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0: stay, 1: left, 2: right, 3: shoot
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # Game variables
        self.bot_position = 0.5  
        self.enemy_position = np.random.uniform(0, 1)
        self.health = 1.0
        self.score = 0
        
        # Pygame setup
        pygame.init()
        self.screen_width = 600
        self.screen_height = 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)

        # Set maximum FPS
        self.max_fps = 60

    def reset(self):
        self.bot_position = 0.5
        self.enemy_position = np.random.uniform(0, 1)
        self.health = 1.0
        self.score = 0
        return np.array([self.bot_position, self.enemy_position, self.health, self.score])

    def step(self, action):
        done = False
        reward = 0

        if action == 1:  # Move left
            self.bot_position = max(0, self.bot_position - 0.1)
        elif action == 2:  # Move right
            self.bot_position = min(1, self.bot_position + 0.1)
        elif action == 3:  # Shoot
            if abs(self.bot_position - self.enemy_position) < 0.1:
                reward = 10  # Hit enemy
                self.enemy_position = np.random.uniform(0, 1)  # Respawn enemy

        # Penalize for missed shot
        if action == 3 and abs(self.bot_position - self.enemy_position) >= 0.1:
            reward -= 1

        # Enemy moves toward the bot
        self.enemy_position -= 0.05
        if self.enemy_position < 0:
            self.health -= 0.1  # Lose health if enemy reaches the player
            self.enemy_position = np.random.uniform(0, 1)  # Respawn enemy
            reward -= 5  # Penalize when enemy passes

        reward += 1  # Bonus for surviving

        if self.health <= 0:
            done = True

        return np.array([self.bot_position, self.enemy_position, self.health, self.score]), reward, done, {}

    def render(self, mode='human'):
        # Fill the background
        self.screen.fill((0, 0, 0))

        # Render the bot (player) as a rectangle
        bot_x = int(self.bot_position * self.screen_width)
        pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(bot_x, self.screen_height - 50, 50, 50))

        # Render the enemy as a rectangle
        enemy_x = int(self.enemy_position * self.screen_width)
        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(enemy_x, 50, 50, 50))

        # Display health and score
        health_text = self.font.render(f'Health: {int(self.health * 100)}%', True, (255, 255, 255))
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(score_text, (10, 50))

        # Update the screen
        pygame.display.flip()
        self.clock.tick(self.max_fps)  # Limit to max FPS

    def close(self):
        pygame.quit()
