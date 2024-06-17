import cv2
from environment import GameEnvironment
from agent import DQNAgent

def record_video(episodes, filename="gameplay.mp4"):
    env = GameEnvironment()
    state_size = env.state.shape[0]
    action_size = 2  # Example: 2 actions (flap or not flap)
    agent = DQNAgent(state_size, action_size)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (600, 400))  # Adjust resolution as needed

    for e in range(episodes):
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            frame = env.render()  # Implement render to return image frame
            out.write(frame)
            if done:
                break

    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_video(10)
