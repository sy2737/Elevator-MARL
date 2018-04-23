import environment as gym
import random
import time
import logging
from Q_learning_ext.nnet import NNet
import tensorflow as tf
import numpy as np

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
    nElevator = 5
    nFloor = 10
    spawnRates = [1/30]+[1/180]*(nFloor-1)
    avgWeight = 135
    weightLimit = 1200
    loadTime = 1
    beta = 0.01
    lr = 1e-4

# initialize environment
env = gym.make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime)
obssize = env.observation_space_size
actsize = env.action_space_size
eval_hours = 3

# initialize tensorflow session
sess = tf.Session()

# optimizer
optimizer = tf.train.AdamOptimizer(lr)

# initialize NNet for each elevator
Q=[]
for i in range(nElevator):
    with tf.variable_scope("Q"+str(i)):
        Q.append(NNet(obssize, actsize, sess, optimizer))

# Use the saver to restore trained model from disk
saver = tf.train.Saver()
saver.restore(sess, "./q_learning_ext_model.ckpt")
print("Model restored.")

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
        prob_dist = np.zeros(actsize)
        q_vals = Q[agent].compute_Qvalues([ss[i]])[0]
        for a in env.legal_actions(agent):
            q_val = q_vals[a] # Qi(s,a)
            prob_dist[a] = np.exp(q_val)
        prob_dist = prob_dist/sum(prob_dist)
        action = np.random.choice(actsize, p = prob_dist)
        actions.append(action)

    env_state_dict = timed_function(env.step)(actions)
    ss = env_state_dict["states"]
    env.render()
    time.sleep(0.5)
