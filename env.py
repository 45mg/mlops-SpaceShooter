import gym
from gym import spaces
import numpy as np
import pygame

class MultiAgentSpaceShooterEnv(gym.Env):
    def __init__(self, num_agents=3, num_enemies=5, max_fps=30, reward_params=None):
        super(MultiAgentSpaceShooterEnv, self).__init__()
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.max_fps = max_fps
        self.reward_params = reward_params if reward_params else {
            'hit_reward': 10,
            'miss_penalty': -0.5,
            'survival_bonus': 2,
            'enemy_pass_penalty': -2,
            'health_loss': 0.01
        }

        # Calculate the correct observation space shape
        obs_shape = (self.num_agents + self.num_enemies + self.num_enemies + self.num_agents + self.num_agents,)
        self.action_space = spaces.MultiDiscrete([4] * self.num_agents)  # 0: stay, 1: left, 2: right, 3: shoot
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

        # Game variables
        self.bot_positions = np.full(self.num_agents, 0.5)
        self.enemy_positions = np.random.uniform(0, 1, self.num_enemies)
        self.enemy_y_positions = np.zeros(self.num_enemies)  # Y positions of enemies
        self.healths = np.ones(self.num_agents)
        self.scores = np.zeros(self.num_agents)
        self.enemy_respawn_times = np.zeros(self.num_enemies)
        self.bullets = []  # List to track bullets

        # Pygame setup
        pygame.init()
        self.screen_width = 800  # Increased playground size
        self.screen_height = 600  # Increased playground size
        self.agent_size = 30  # Decreased agent size
        self.enemy_size = 30  # Decreased enemy size
        self.bullet_size = 5  # Size of bullets
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)

    def reset(self):
        self.bot_positions = np.full(self.num_agents, 0.5)
        self.enemy_positions = np.random.uniform(0, 1, self.num_enemies)
        self.enemy_y_positions = np.zeros(self.num_enemies)  # Reset Y positions of enemies
        self.healths = np.ones(self.num_agents)
        self.scores = np.zeros(self.num_agents)
        self.enemy_respawn_times = np.zeros(self.num_enemies)
        self.bullets = []  # Reset bullets
        return self._get_obs()

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        done = False

        for i, action in enumerate(actions):
            if action == 1:  # Move left
                self.bot_positions[i] = max(0, self.bot_positions[i] - 0.1)
            elif action == 2:  # Move right
                self.bot_positions[i] = min(1, self.bot_positions[i] + 0.1)
            elif action == 3:  # Shoot
                self.bullets.append([self.bot_positions[i], 1.0])  # Add bullet at agent's position

        # Move bullets upwards
        new_bullets = []
        for bullet in self.bullets:
            bullet[1] -= 0.1  # Move bullet up
            if bullet[1] > 0:
                new_bullets.append(bullet)
        self.bullets = new_bullets

        # Check for bullet collisions with enemies
        for bullet in self.bullets:
            for j, (enemy_x, enemy_y) in enumerate(zip(self.enemy_positions, self.enemy_y_positions)):
                if abs(bullet[0] - enemy_x) < 0.05 and abs(bullet[1] - enemy_y) < 0.05:
                    rewards += self.reward_params['hit_reward']  # Hit enemy
                    self.scores += 1  # Increase score
                    self.enemy_respawn_times[j] = 50  # Set respawn time
                    self.enemy_positions[j] = np.random.uniform(0, 1)  # Respawn enemy
                    self.enemy_y_positions[j] = 0  # Reset Y position
                    self.bullets.remove(bullet)  # Remove bullet
                    break

        # Enemy moves toward the bots
        self.enemy_y_positions += 0.01  # Move enemies downwards slowly
        for j, (enemy_x, enemy_y) in enumerate(zip(self.enemy_positions, self.enemy_y_positions)):
            if enemy_y >= 1:
                for i in range(self.num_agents):
                    self.healths[i] -= self.reward_params['health_loss']  # Lose health if enemy reaches the player
                    rewards[i] += self.reward_params['enemy_pass_penalty']  # Penalize when enemy passes
                self.enemy_positions[j] = np.random.uniform(0, 1)  # Respawn enemy
                self.enemy_y_positions[j] = 0  # Reset Y position

        # Handle enemy respawn
        for j in range(self.num_enemies):
            if self.enemy_respawn_times[j] > 0:
                self.enemy_respawn_times[j] -= 1
                if self.enemy_respawn_times[j] == 0:
                    self.enemy_positions[j] = np.random.uniform(0, 1)  # Respawn enemy
                    self.enemy_y_positions[j] = 0  # Reset Y position

        rewards += self.reward_params['survival_bonus']  # Bonus for surviving

        if np.any(self.healths <= 0):
            done = True

        # Sum the rewards to return a single scalar reward
        total_reward = np.sum(rewards)

        return self._get_obs(), total_reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.bot_positions, self.enemy_positions, self.enemy_y_positions, self.healths, self.scores])

    def render(self, mode='human'):
        # Fill the background
        self.screen.fill((0, 0, 0))

        # Render the bots (players) as rectangles
        for bot_pos in self.bot_positions:
            bot_x = int(bot_pos * self.screen_width)
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(bot_x, self.screen_height - self.agent_size, self.agent_size, self.agent_size))

        # Render the enemies as rectangles
        for enemy_x, enemy_y in zip(self.enemy_positions, self.enemy_y_positions):
            enemy_x = int(enemy_x * self.screen_width)
            enemy_y = int(enemy_y * self.screen_height)
            pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(enemy_x, enemy_y, self.enemy_size, self.enemy_size))

        # Render the bullets as rectangles
        for bullet in self.bullets:
            bullet_x = int(bullet[0] * self.screen_width)
            bullet_y = int(bullet[1] * self.screen_height)
            pygame.draw.rect(self.screen, (255, 255, 0), pygame.Rect(bullet_x, bullet_y, self.bullet_size, self.bullet_size))

        # Display health and score for each agent
        for i in range(self.num_agents):
            health_text = self.font.render(f'Agent {i+1} Health: {int(self.healths[i] * 100)}%', True, (255, 255, 255))
            score_text = self.font.render(f'Agent {i+1} Score: {int(self.scores[i])}', True, (255, 255, 255))
            self.screen.blit(health_text, (10, 10 + i * 40))
            self.screen.blit(score_text, (10, 30 + i * 40))

        # Update the screen
        pygame.display.flip()
        self.clock.tick(self.max_fps)  # Limit to max FPS

    def close(self):
        pygame.quit()