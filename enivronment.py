import numpy as np
import pygame

class GameEnvironment:
    def __init__(self):
        self.width = 600
        self.height = 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.state = np.zeros((4,))  # Example: a state with 4 features
        self.player_pos = [self.width // 2, self.height // 2]
        self.done = False
        return self.state

    def step(self, action):
        if action == 0:  # Action 0: Move up
            self.player_pos[1] -= 10
        elif action == 1:  # Action 1: Move down
            self.player_pos[1] += 10

        # Update the state (for example purposes, state is just player position)
        self.state = np.array(self.player_pos + [0, 0])

        # Check if game is over (for example purposes, hitting the screen boundaries)
        if self.player_pos[1] < 0 or self.player_pos[1] > self.height:
            self.done = True
            reward = -1
        else:
            reward = 1
        
        next_state = self.state
        return next_state, reward, self.done

    def render(self):
        self.screen.fill((0, 0, 0))  # Clear screen with black
        pygame.draw.rect(self.screen, (0, 128, 255), pygame.Rect(self.player_pos[0], self.player_pos[1], 50, 50))  # Draw player
        pygame.display.flip()  # Update the display
        self.clock.tick(30)  # Limit to 30 FPS

    def close(self):
        pygame.quit()

# Example usage
if __name__ == "__main__":
    pygame.init()
    env = GameEnvironment()
    state = env.reset()
    for _ in range(100):
        action = np.random.choice([0, 1])  # Random action for demonstration
        next_state, reward, done = env.step(action)
        env.render()
        if done:
            state = env.reset()
    env.close()
