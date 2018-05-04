# environment returns observation size and action size
obssize = env.observation_space.size
actsize = env.action_space.n

# environment returns current time
env.now()


'''
get_states return a dictionary also with
"state": actual_state,
"R": cumul_cost_list,
"time_elapsed": time_elapsed_list

where time_elapsed_list has length nElevator,
and only decision elevators' corresponding index has nonzero values
that correspond to the time elapsed since its last decision epoch
if it's the first time the decision elevator i performs action,
time_elapsed_list[i] = None
'''
