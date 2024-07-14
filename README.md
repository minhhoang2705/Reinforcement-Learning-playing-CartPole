# Reinforcement Learning Framework from Scratch

This repository contains an implementation of a Reinforcement Learning framework from scratch, applied to a simple game environment.

## Project Structure

```
RL/
|-- test_agent.py
|-- train_agent.py
|-- visualizer.py
|-- README.md
```

- **train_agent.py:** Contains the DQNAgent class, which defines the agent's behavior and learning process.
- **test_agent.py:** Contains the test and demo for the DQN agent.
- **visualizer.py:** Plot rewards and losses for the agent.

## Setup Instructions

### Prerequisites
- Python 3.x
- Libraries: tensorflow, gym, pygame, matplotlib

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/RLFramework.git
    cd RLFramework
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

### Training the Agent
To train the DQN agent, run the following command:
```
python train_agent.py
```

### Recording Gameplay
To record the agent's gameplay, run the follwoing command:
```
python test_agent.py
```

## Result 
After training, a plot of the total rewards per episode will be displayedm showing the learning progress of the agent.

## Demo Video
A demo video of the agent playing the game can be found here [here](https://drive.google.com/file/d/14wACXJGy5b452vsfVd6MLeeF2h4Hyh7N/view?usp=drivesdk)

## License
