import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from collections import deque

env = gym.make("ALE/Assault-v5" , render_mode="human")
learning_rate = 0.001
discount_factor = 0.95
num_episodes = 1000
max_steps = 1000
preprocess_space = (84, 84)
replay_buffer_size = 1000
batch_size = 64

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(*preprocess_space, 1)),
    keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(env.action_space.n)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss='mse') 
print(model.summary())

def preprocess_observation(observation):
    gray_image = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(gray_image, preprocess_space, interpolation=cv2.INTER_AREA)
    return resized_image.reshape((*preprocess_space, 1))  # Ensure the input shape matches the model

replay_buffer = deque(maxlen=replay_buffer_size)


def add_to_replay_buffer(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

episode_rewards = []
plt.figure(figsize=(12, 6))

for episode in range(num_episodes):
    state = env.reset() 
    state = state[0]
    state = preprocess_observation(state)
    episode_reward = 0

    for step in range(max_steps):
        if np.random.uniform(0, 1) < 0.1:
            action = env.action_space.sample()  
        else:
            q_values = model.predict(np.expand_dims(state, axis=0))
            action = np.argmax(q_values[0])

        env.render()
        new_state, reward, done, _, _ = env.step(action)
        new_state = preprocess_observation(new_state)
        episode_reward += reward

        add_to_replay_buffer(state, action, reward, new_state, done)

        if len(replay_buffer) >= batch_size:
            batch_indices = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
            batch = [replay_buffer[idx] for idx in batch_indices]

            states = np.array([transition[0] for transition in batch])
            actions = np.array([transition[1] for transition in batch]) 
            rewards = np.array([transition[2] for transition in batch])
            next_states = np.array([transition[3] for transition in batch])
            dones = np.array([transition[4] for transition in batch])

            
            targets = rewards + (1 - dones) * discount_factor * np.max(model.predict(next_states), axis=1)
            target_f = model.predict(states)
            target_f[np.arange(batch_size), actions] = targets
            
            model.fit(states, target_f, epochs=1, verbose=0)

        state = new_state

        if done:
            break   

    episode_rewards.append(episode_reward)

plt.show()



# Close the environment after training
env.close()
