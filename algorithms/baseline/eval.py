import sys
sys.path.insert(0, '/Users/shatianwang/Desktop/Elevator-MARL')
import environment as gym
import random
import time
import logging
from Q_learning_ext.nnet import NNet
import tensorflow as tf
import numpy as np


"""
Gather command line input
"""
if len(sys.argv) != 4:
    print("Input Format: python training.py nElevator(int) nFloor(int) inputFilePath(str)")
    exit(1)
print(" nElevator = ", sys.argv[1], "\n",
      "nFloor = ", sys.argv[2], "\n",
      "Input File Path = ", sys.argv[3])

answer = input('Starting testing?: [y/n]')
if not answer or answer[0].lower() != 'y':
    exit(1)

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
    logging.disable(logging.NOTSET)
    nElevator = int(sys.argv[1])
    nFloor = int(sys.argv[2])
    spawnRates = [1/360]+[1/360]*(nFloor-1)
    avgWeight = 135
    weightLimit = 1200
    loadTime = 1
    beta = 0.01
    lr = 1e-4

"""
Initialize environment and optimizers
"""
# initialize environment
env = gym.make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime)
obssize = env.observation_space_size
actsize = env.action_space_size
print("state space dimension", obssize)
print("action space size", actsize)

# initialize tensorflow session
sess = tf.Session()

# initialize an optimizer for each elevator
optimizer_list = []
for i in range(nElevator):
    optimizer_list.append(tf.train.AdamOptimizer(lr))

# initialize a NNet for each elevator
Q=[]
for i in range(nElevator):
    with tf.variable_scope("Q"+str(i)):
        Q.append(NNet(obssize, actsize, sess, optimizer_list[i]))

# if train using existing network
input_ckpt_path = sys.argv[3] # specify path to the checkpoint file
saver = tf.train.Saver()
saver.restore(sess, input_ckpt_path)
print("Model restored.")

# initialize environment
env = gym.make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime)
obssize = env.observation_space_size
actsize = env.action_space_size
# specify number of hours to evaluate model
eval_hours = 30

"""
Begin testing
"""
env_state_dict = env.reset()
ss = env_state_dict["states"]

# main iteration of training
# env.now() gives the total elapsed simulation time in seconds
while env.now() <= eval_hours * 3600:
    print("current time is: -----", env.now(), "-----")
    decision_agents = env_state_dict["decision agents"]
    actions = []
    for i in range(len(decision_agents)):
        agent = decision_agents[i]
        # agent makes an action choice
        legal_actions_bool = np.full(actsize, False)
        for action in env.legal_actions(agent):
            legal_actions_bool[action] = True
        prob_dist, _ = Q[agent].compute_legal_action_prob_dist([ss[i]], legal_actions_bool, 0.000001)
        print(prob_dist)

        legal_actions_list = list(env.legal_actions(agent))
        legal_actions_list.sort()
        action = np.random.choice(legal_actions_list, p = prob_dist)
        # update prev_actions and prev_states lists
        actions.append(action)
        # print("state representation: ", ss[i])
    for i in range(len(decision_agents)):
        print("decision agent ", decision_agents[i])
        print("action to take: ", actions[i])
        print("requested calls from within: ", ss[i][-1-nFloor:-1])
    if env.nPassenger_served >= 1:
        print("avg wait time is: ", env.avg_wait_time())
        # print("reward: ", env_state_dict["rewards"][i])

    env_state_dict = timed_function(env.step)(actions)
    ss = env_state_dict["states"]

    env.render()
    time.sleep(0.5)
