import gym 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt

custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=False , render_mode="human")
learning_rate = 0.001
num_episodes = 1000
max_steps = 50
initial_epsilon = 1
epsilon_decay = 0.995
gamma = 0.95

model = keras.models.Sequential([
    keras.layers.Dense(10,activation = "relu" ),
    keras.layers.Dense(env.action_space.n , activation = "relu")
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate) , loss= "mse")
print(model.summary())


for episode in range(num_episodes):
    print("start")
    initial_step = env.reset()
    state = state = np.array([initial_step[0]])
    epsilon = initial_epsilon * epsilon_decay ** episode
    episode_rewards = []

    for step in range(max_steps):
        print("step")
        episode_reward = 0
        if np.random.uniform(0,1)<epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.expand_dims(state , 0)))

        step_result = env.step(action)
        env.render()
        next_state = np.array([step_result[0]])
        fell = step_result[2]
        ep_end = step_result[3]
        if fell:
            reward = -1
        else:
            reward = step_result[0]

        delta = reward + (1-ep_end)*gamma*(np.max(model.predict(np.expand_dims(next_state , 0))))
        delta = np.expand_dims(delta , 0) # shape = (1, 1)

        target = model.predict(np.expand_dims(state , 0)) # shape = (1 , n)
        target[0 , action ] = delta

        model.fit(state , target , epochs = 1 , verbose = 1)
        episode_reward+=reward

        if ep_end or fell:
            break
        state = next_state

    episode_rewards.append(episode_reward)
    print(episode_rewards)

env.close()