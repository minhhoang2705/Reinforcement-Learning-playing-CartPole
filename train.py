import numpy as np
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

    for e in range(episodes):
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(replay_buffer) > batch_size:
                agent.replay(batch_size)

if __name__ == "__main__":
    train_dqn(1000)
