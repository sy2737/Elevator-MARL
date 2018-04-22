import environment as gym
import numpy as np
import random
import time
import logging




logger = gym.logger.get_my_logger(__name__)

def timed_function(func):
    def decorated_func(*args, **kwargs):
        s = time.time()
        output = func(*args, **kwargs)
        e = time.time()
        logger.info("{} finished in {} seconds".format(func.__name__,e-s))
        return output
    return decorated_func

def pretty_print_states(state, nFloor, nElevator, nStates=3):
    hall_calls_up = state[0:nFloor]
    hall_calls_down = state[nFloor:nFloor*2]
    hall_call_up_times = state[nFloor*2:nFloor*3]
    hall_call_down_times = state[nFloor*3:nFloor*4]
    onehot_elevator_positions = state[nFloor*4:(nFloor*4+nElevator*nFloor)].reshape(nElevator,nFloor)
    onehot_elevator_states = state[(nFloor*4+nElevator*nFloor):-1]
    time_elapsed = state[-1]
    logger.info("Hall Calls UP: {}".format(hall_call_up_times))
    logger.info("Hall Calls DOWN: {}".format(hall_call_down_times))

    #print("state representation as seen from onehot encoding of positions")
    #print(np.array([row[::-1] for row in onehot_elevator_positions]).T)


if __name__=="__main__":
    logging.disable(logging.DEBUG)
    nElevator = 5
    nFloor = 20
    spawnRates = [1/30]+[1/180]*(nFloor-1)
    avgWeight = 135
    weightLimit = 1200
    loadTime = 1
    

    env = gym.make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime)
    s = env.reset()
    while True:
        decision_agents = s["decision agents"]
        states = s["states"]
        rewards = s["rewards"]
        logger.info("Rewards of decision elevators: {}".format(rewards))
        pretty_print_states(states[0], nFloor, nElevator)
        actions = [random.sample(env.legal_actions(agent), 1)[0] for agent in decision_agents]# 0 calls for the legal actions of the first elevator
        #actions = [6]*len(decision_agents)

        s = timed_function(env.step)(actions)
        env.render()
        time.sleep(1)
