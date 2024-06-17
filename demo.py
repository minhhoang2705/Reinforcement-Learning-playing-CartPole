import cv2
import numpy as np
import pygame
from agent import DQNAgent
from environment import GameEnvironment

def record_video(episodes, filename="gameplay.mp4"):
    env = GameEnvironment()
    state_size = env.state.shape[0]
    action_size = 2  # Example: 2 actions (flap or not flap)
    agent = DQNAgent(state_size, action_size)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (env.width, env.height))  # Adjust resolution as needed

    for e in range(episodes):
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state

            # Render the game to a surface and capture the pixels
            env.render()
            pixels = pygame.surfarray.array3d(env.screen)
            frame = cv2.cvtColor(np.transpose(pixels, (1, 0, 2)), cv2.COLOR_RGB2BGR)
            out.write(frame)

            if done:
                break

    out.release()
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pygame.init()
    record_video(10)
    pygame.quit()
