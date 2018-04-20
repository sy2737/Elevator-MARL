from env import make
from elevator import Elevator
import random
import time




def timed_function(func):
    def decorated_func(*args, **kwargs):
        s = time.time()
        output = func(*args, **kwargs)
        e = time.time()
        print(func.__name__, "finished in {} seconds".format(e-s))
        return output
    return decorated_func



if __name__=="__main__":
    env = make(2, 10, [1/180]*10, 135, 1200, 5, 1)
    s = env.reset()
    while True:
        decision_agent = s["decision_elevator"]
        action = random.sample(env.legal_actions(decision_agent), 1)[0]# 0 calls for the legal actions of the first elevator
        print("Taking action {}:{}".format(action, env.elevators[0].ACTION_FUNCTION_MAP[action].__name__))
        s = timed_function(env.step)(action)

        env.render()
        #if s['elevator_positions'][0] in [0, 9]:
        #    if action == 3:
        #        action = 5
        #    else:
        #        action = 3
        time.sleep(0.5)
