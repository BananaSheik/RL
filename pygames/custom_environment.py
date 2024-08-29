import gym
from gym import spaces
import pygame
import numpy as np
import random

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  
        self.window_size = 512 

        self.observation_space = spaces.Dict({
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            })

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        if terminated :
            reward = 1
        elif self._agent_location[0]<=0 or self._agent_location[0]>=self.size-1 or self._agent_location[1]<=0 or self._agent_location[1]>=self.size-1:
            reward = -1
        else:
            reward =1
         
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (  self.window_size / self.size   )  

        pygame.draw.rect(canvas,(255, 0, 0) , pygame.Rect( pix_square_size * self._target_location,(pix_square_size, pix_square_size),),  )
        pygame.draw.circle(canvas,(0, 0, 255),(self._agent_location + 0.5) * pix_square_size,pix_square_size / 3,)

        for x in range(self.size + 1):
            pygame.draw.line(canvas,0,(0, pix_square_size * x),(self.window_size, pix_square_size * x),width=3,)
            pygame.draw.line(canvas,0,(pix_square_size * x, 0),(pix_square_size * x, self.window_size),width=3,)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

def generate_episode(env, policy , epsilon):
    # episode = []
    # state, _ = env.reset()
    # terminated = False
    # while not terminated:
    #     action = policy[state["agent"][0], state["agent"][1]]
    #     next_state, reward, terminated, _, info = env.step(action)
    #     episode.append((state, action, reward))
    #     state = next_state
    # return episode
    episode = []
    obs , info = env.reset()
    terminated = False
    state = tuple(obs["agent"])

    while not terminated:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = policy[state]

        next_state , reward , terminated , _ , info = env.step(action)
        episode.append((state , action , reward))
        state = tuple(next_state["agent"])

    return episode


def monte_carlo_control(env, num_episodes, gamma=1 , initial_epsilon = 1 , epsilon_decay=0.99, epsilon_min=0.01):
    returns = {}
    Q = np.ones((env.size, env.size, env.action_space.n))
    policy = {}
    for x in range(env.size):
        for y in range(env.size):
            policy[(x,y)] = random.randint(0 ,env.action_space.n-1)
    
    for episode_num in range(num_episodes):
        epsilon = max(epsilon_min, initial_epsilon * (epsilon_decay ** episode_num))
        
        episode = generate_episode(env, policy , epsilon)
        G = 0
        for state , action , reward  in reversed(episode[:]):
            G = gamma * G + reward
            state_tuple = tuple(state)
            if (state_tuple , action) not in returns:
                returns[(state_tuple , action)] = []
                
            returns[(state_tuple , action)].append(G)
            Q[state_tuple[0], state_tuple[1], action] = np.mean(returns[(state_tuple, action)])
            policy[state_tuple] = np.argmax(Q[state_tuple[0], state_tuple[1]])

        print(Q)
        print(policy)
        print("next")
    return policy, Q



env = GridWorldEnv(render_mode="human", size=5)
policy, Q = monte_carlo_control(env, num_episodes=5000)

print("Learned policy:")
print(policy)
