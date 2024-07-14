# visualize_training.py
import matplotlib.pyplot as plt
import json

# Read rewards and losses from the JSON file
with open("training_log.json", "r") as f:
    log = json.load(f)

episode_rewards = log["episode_rewards"]
episode_losses = log["episode_losses"]

# Plot rewards and losses
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(episode_rewards, label='Episode Reward')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.set_title('Episode Rewards Over Time')
ax1.legend()

ax2.plot(episode_losses, label='Training Loss', color='r')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss Over Time')
ax2.legend()

plt.tight_layout()
plt.show()
