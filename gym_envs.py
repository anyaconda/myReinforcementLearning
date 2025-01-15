#meta 1/7/2025 myReinforcement Learning

#References
#https://www.gymlibrary.dev/
#https://www.gymlibrary.dev/environments/classic_control/
#  See cart_pole0.py

#---------------------------------------------------------------------------------------------------------
import gym
from gym import envs
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
#view available envs
print(envs.registry.keys())
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
#env = gym.make("Blackjack-v1", render_mode="human")
env = gym.make("FrozenLake-v1", render_mode="human")

#code from cart_pole0.py exactly to start with, probably doesn't do anything until coded for specific game
# Reset the environment to start
state = env.reset()

N_STEPS = 100

# Run for n timesteps
for _ in range(N_STEPS):
    env.render()  # Render the environment
    action = env.action_space.sample()  # Take a random action
    state, reward, done, info = env.step(action)[:4]  # Step the environment by one timestep #$acdelta

    # If the episode is done (CartPole has fallen), reset the environment
    print(_, state, reward, done, info, action)
    if done:
        state = env.reset()
        #break

env.close() 
#---------------------------------------------------------------------------------------------------------