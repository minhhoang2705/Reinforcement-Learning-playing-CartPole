import numpy as np
import matplotlib.pyplot as plt
from agent import DQNAgent
from environment import GameEnvironment
from replay_buffer import ReplayBuffer

def train_dqn(episodes):
    env = GameEnvironment()
    state_size = env.state.shape[0]
    action_size = 2  # Example: 2 actions (flap or not flap)
    agent = DQNAgent(state_size, action_size)
    replay_buffer = ReplayBuffer(2000)
    batch_size = 32
    rewards = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                rewards.append(total_reward)
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(replay_buffer) > batch_size:
                agent.replay(batch_size)

    plt.plot(rewards)
    plt.ylabel('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.show()

if __name__ == "__main__":
    train_dqn(1000)
