Use the make function in env.py to construct environments.

Currently there are two types of reward structure.  
**Version 0:** the reward is accumulated with a rate equal to the sum of squared waiting time of the passengers. (So technically this should be considered as loss. Therefore if this environment is used, you need to negate the sign of the reward manually)  
**Version 1:** the reward is accumulated with a rate of -1 for as long as there are passengers in the system. So for example if there are at least one passengers somewhere in the system, and elevator 0's last decision epoch was 10 seconds ago, then in its' current decision epoch it observes -10. 

