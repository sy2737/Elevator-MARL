import environment as gym
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
        decision_agents = s["decision_elevators"]
        actions = [random.sample(env.legal_actions(agent), 1)[0] for agent in decision_agents]# 0 calls for the legal actions of the first elevator
        #actions = [6]*len(decision_agents)

        s = timed_function(env.step)(actions)
        env.render()
        #time.sleep(0.2)
