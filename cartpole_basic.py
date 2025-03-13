#meta 1/6/2025 myReinforcement Learning

#infra on-prem: laptop + VSCode 
#      env: .venv in anya-rl
#      confirmed Python 3.10.4
#      downgraded numpy 1.23.5
#      downgraded gym 0.25.2, pygame 2.1.0
#      pip 24.3.1
#      no ipykernel, ipython

#References
#  Reinforcement Learning: An Overview refer to https://www.geeksforgeeks.org/what-is-reinforcement-learning/

#---------------------------------------------------------------------------------------------------------
import gym
import numpy as np
##import warnings
#---------------------------------------------------------------------------------------------------------

# Suppress specific deprecation warnings
##warnings.filterwarnings("ignore", category=DeprecationWarning)

#---------------------------------------------------------------------------------------------------------
# Load the environment with render mode specified
env = gym.make('CartPole-v1', render_mode="human")

# Initialize the environment to get the initial state
state = env.reset()

# Print the state space and action space
print("State space:", env.observation_space)
print("Action space:", env.action_space)

# Run a few steps in the environment with random actions
for _ in range(100):
    env.render()  # Render the environment for visualization
    action = env.action_space.sample()  # Take a random action
    
    # Take a step in the environment
    step_result = env.step(action)
    
    # Check the number of values returned and unpack accordingly
    if len(step_result) == 4:
        next_state, reward, done, info = step_result
        terminated = False
    else:
        next_state, reward, done, truncated, info = step_result
        terminated = done or truncated
    
    print(f"Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}, Info: {info}")
    
    if terminated:
        state = env.reset()  # Reset the environment if the episode is finished

env.close()  # Close the environment when done
#---------------------------------------------------------------------------------------------------------