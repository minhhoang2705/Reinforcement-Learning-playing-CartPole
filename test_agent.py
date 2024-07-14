import gym

import keras
import numpy as np
from keras.saving import register_keras_serializable

env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
state_size = env.observation_space.shape[0]


@register_keras_serializable()
def mse(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred)


my_agent = keras.models.load_model("train_agent.h5", custom_objects={"mse": mse})
n_timesteps = 500
total_reward = 0

for t in range(n_timesteps):
    env.render()
    state = state.reshape((1, state_size))
    q_values = my_agent.predict(state, verbose=0)
    max_q_values = np.argmax(q_values)

    next_state, reward, terminal, truncated, _ = env.step(action=max_q_values)
    done = terminal or truncated
    total_reward += reward
    state = next_state
    print(t)
    if done:
        break

env.close()
print("Total reward = ", total_reward)
