import random
import numpy as np
import gym
import tensorflow
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class MyAgent:
    def __init__(self, state_size, action_size):
        """
        Initializes the MyAgent object.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The number of actions available.

        Initializes the following instance variables:
            - state_size (int): The size of the state space.
            - action_size (int): The number of actions available.
            - replay_buffer (deque): A deque to store experiences.
            - gamma (float): The discount factor.
            - epsilon (float): The initial value of epsilon for epsilon-greedy exploration.
            - epsilon_min (float): The minimum value of epsilon.
            - epsilon_decay (float): The decay rate for epsilon.
            - learning_rate (float): The learning rate for the neural network.
            - update_targetnn_rate (int): The rate at which the target network is updated.
            - main_network (nn.Module): The main neural network.
            - target_network (nn.Module): The target neural network.

        Sets the weights of the target network to be the same as the main network.
        """
        self.state_size = state_size
        self.action_size = action_size

        self.replay_buffer = deque(maxlen=50000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_targetnn_rate = 10

        self.main_network = self._create_neural_network()
        self.target_network = self._create_neural_network()

        self.target_network.set_weights(self.main_network.get_weights())

    def _create_neural_network(self):
        """
        Creates and compiles a neural network model using the Keras library.

        Returns:
            model (Sequential): The compiled neural network model.
        """
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save_experience(self, state, action, reward, next_state, terminal):
        """
        Saves the agent's experience (state, action, reward, next_state, terminal) into the replay buffer.

        Parameters:
            state (object): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (object): The next state of the environment.
            terminal (bool): Indicates if the episode has terminated.

        Returns:
            None
        """
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def get_batch_from_buffer(self, batch_size):
        """
        Retrieves a batch of experiences from the replay buffer.

        Parameters:
            batch_size (int): The number of experiences to retrieve.

        Returns:
            tuple: A tuple containing the following:
                - state_batch (ndarray): An array of shape (batch_size, state_size) containing the states of the experiences.
                - action_batch (ndarray): An array of shape (batch_size,) containing the actions taken in the experiences.
                - reward_batch (list): A list of floats containing the rewards received after taking the actions in the experiences.
                - next_state_batch (ndarray): An array of shape (batch_size, state_size) containing the next states of the experiences.
                - terminal_batch (list): A list of booleans indicating whether the experiences have terminated.

        This function retrieves a random sample of experiences from the replay buffer. 
        The sample size is determined by the `batch_size` parameter. 
        The function then extracts the states, actions, rewards, next states, and terminal flags from each experience in the sample. 
        The states and next states are stored in arrays of shape (batch_size, state_size). 
        The actions and terminal flags are stored in arrays of shape (batch_size,). 
        The function returns these arrays as a tuple.
        """
        # Get a random sample of experiences from the replay buffer
        exp_batch = random.sample(self.replay_buffer, batch_size)

        # Extract the states, actions, rewards, next states, and terminal flags from each experience
        state_batch = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size)
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size)
        terminal_batch = [batch[4] for batch in exp_batch]

        # Return the extracted arrays as a tuple
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def train_main_network(self, batch_size):
        """
        Trains the main network using a batch of experiences from the replay buffer.

        Parameters:
            batch_size (int): The number of experiences to use for training.

        Returns:
            None

        This function is responsible for training the main network using a batch of experiences from the replay buffer.
        It follows the Q-learning algorithm to update the Q-values for the actions taken in each experience.

        Steps involved in training the main network:
        1. Retrieve a batch of experiences from the replay buffer using the `get_batch_from_buffer` method.
        2. Predict the Q-values for the states in the batch using the `predict` method of the main network.
        3. Predict the Q-values for the next states in the batch using the `predict` method of the target network.
        4. Calculate the maximum Q-value for each next state using `np.amax`.
        5. For each experience in the batch, update the Q-value for the action taken based on the reward and the maximum Q-value for the next state.
        6. Train the main network using the states in the batch and the updated Q-values using the `fit` method.

        The purpose of training the main network is to update its Q-values so that it can make better decisions in the future.
        """
        # Retrieve a batch of experiences from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(batch_size)

        # Predict the Q-values for the states in the batch using the main network
        q_values = self.main_network.predict(state_batch, verbose=0)

        # Predict the Q-values for the next states in the batch using the target network
        next_q_values = self.target_network.predict(next_state_batch, verbose=0)

        # Calculate the maximum Q-value for each next state
        max_next_q = np.amax(next_q_values, axis=1)

        # Update the Q-values for each experience based on the reward and the maximum Q-value for the next state
        for i in range(batch_size):
            if terminal_batch[i]:
                # If the experience has terminated, set the Q-value for the action taken to the reward received
                new_q_values = reward_batch[i]
            else:
                # Otherwise, set the Q-value for the action taken to the reward received plus the discounted maximum Q-value for the next state
                new_q_values = reward_batch[i] + self.gamma * max_next_q[i]
            # Update the Q-value for the action taken in the experience
            q_values[i][action_batch[i]] = new_q_values

        # Train the main network using the states in the batch and the updated Q-values
        self.main_network.fit(state_batch, q_values, verbose=0)

    def make_decision(self, state):
        """
        Makes a decision based on the given state.

        Parameters:
            state: The current state of the agent.

        Returns:
            int: The action to take based on the Q-values predicted by the main network.
        """
        # If the agent is exploring, choose a random action with probability self.epsilon
        # Otherwise, choose the action with the highest Q-value
        if random.uniform(0,1) < self.epsilon:
            # Choose a random action
            return np.random.randint(self.action_size)
        else:
            # Reshape the state to be a 2D array of shape (1, self.state_size)
            state = state.reshape((1, self.state_size))
            # Predict the Q-values for the state using the main network
            q_values = self.main_network.predict(state, verbose=0)
            # Return the action with the highest Q-value
            return np.argmax(q_values[0])

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state, _ = env.reset()

    state_size = env.observation_space.shape[0] # retrieve cart position
    action_size = env.action_space.n # 2 actions: left or right

    n_episodes = 100
    n_timesteps = 500
    batch_size = 64

    my_agent = MyAgent(state_size, action_size)
    total_time_step = 0

    # Train the agent for a specified number of episodes
    for ep in range(n_episodes):

        # Initialize the episode reward and reset the environment
        ep_rewards = 0
        state, _ = env.reset()

        # Train the agent for a specified number of timesteps within each episode
        for t in range(n_timesteps):

            # Update the target network every specified number of timesteps
            total_time_step += 1
            if total_time_step % my_agent.update_targetnn_rate == 0:
                my_agent.target_network.set_weights(my_agent.main_network.get_weights())

            # Make a decision based on the current state
            action = my_agent.make_decision(
                state)

            # Take the action, observe the next state, reward, and terminal status
            next_state, reward, terminal, _, _ = env.step(action)

            # Save the experience (state, action, reward, next_state, terminal)
            my_agent.save_experience(state,action, reward, next_state, terminal)

            # Update the current state
            state = next_state

            # Update the episode reward
            ep_rewards += reward

            # If the episode has terminated, print the episode number and reward
            if terminal:
                print("Ep ", ep+1, " reach terminal with reward = ", ep_rewards)
                break

            # If the replay buffer has enough experiences, train the main network
            if len(my_agent.replay_buffer) > batch_size:
                my_agent.train_main_network(batch_size)

        # Decay the value of epsilon (exploration rate) gradually
        if my_agent.epsilon > my_agent.epsilon_min:
            my_agent.epsilon = my_agent.epsilon * my_agent.epsilon_decay

    # Save weights
    my_agent.main_network.save("train_agent.h5")
