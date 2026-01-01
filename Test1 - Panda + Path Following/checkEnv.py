from pandaFollowTaskEnv import pandaFollowTaskEnv
import time


env = pandaFollowTaskEnv(render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()                                      # random action
    print(f"Action : {action}")
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.075)
    print(f"Observation : {observation}")
    print(f"Reward : {reward}")
    print(f"Terminated : {terminated}")
    print(f"Truncated : {truncated}")
    if terminated or truncated:
        observation, info = env.reset()