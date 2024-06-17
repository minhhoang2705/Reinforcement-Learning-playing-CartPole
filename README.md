# Reinforcement Learning Framework from Scratch

This repository contains an implementation of a Reinforcement Learning framework from scratch, applied to a simple game environment.

## Project Structure

```
RL/
|-- agent.py
|-- environment.py
|-- replay_buffer.py
|-- train.py
|-- main.py
|-- demo.py
|-- README.md
```

- **agent.py:** Contains the DQNAgent class, which defines the agent's behavior and learning process.
- **environment.py:** Contains the GameEnvironment class, which simulates the game environment.
- **replay_buffer.py:** Contains the ReplayBuffer class, which stores the agent's experiences.
- **train.py:** Contains the training loop for the DQN agent.
- **main.py:** Entry point for training the agent.
- **demo.py:** Script to record gameplay video.

## Setup Instructions

### Prerequisites
- Python 3.x
- Libraries: numpy, torch, matplotlib, opencv-python

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/RLFramework.git
    cd RLFramework
    ```

2. Install the required libraries:
    ```bash
    pip install numpy torch matplotlib opencv-python
    ```

## Running the Code

### Training the Agent
To train the DQN agent, run the following command:
```
python main.py
```

### Recording Gameplay
To record the agent's gameplay, run the follwoing command:
```
python demo.py
```

## Result 
After training, a plot of the total rewards per episode will be displayedm showing the learning progress of the agent.

## Demo Video
A demo video of the agent playing the game can be found here [here](https://drive.google.com/file/d/14wACXJGy5b452vsfVd6MLeeF2h4Hyh7N/view?usp=drivesdk)

## License
