Model: DDPG-1752096659

Change:
Refreseh

Reward: 
Normaliztion factor --> Num points * 10 + (timesteps - Numpoints)

    1. Waypoint Reward = (flag_index + 1) * 10 

    2. Dwell Reward = +1 for each step on goal

    Reward = (waypoint + dwell) * 1/Normalization factor 
                                                                                      


Total Steps : 50000 * 20 ---> 5000 episodes
Envs Used   : 16 (IGMR PC)


Results:
Simulation:
Forgot to fix the singularity problem



Updates for Next Run: 

