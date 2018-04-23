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
    logging.disable(logging.DEBUG)
    nElevator = 5
    nFloor = 10
    spawnRates = [1/30]+[1/180]*(nFloor-1)
    avgWeight = 135
    weightLimit = 1200
    loadTime = 1
    lr = 0.4*1e-3
    beta = 0.01

# initialize environment
env = gym.make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime)
obssize = env.observation_space_size # Unimplemented in env
actsize = env.action_space_size # Unimplemented in env
simulation_hours = 3750
factor = 0.998

# initialize tensorflow session
sess = tf.Session()

# optimizer
optimizer = tf.train.AdamOptimizer(lr)

# initialize NNet for each elevator
Q=[]
for i in range(nElevator):
    with tf.variable_scope("Q"+str(i)):
        Q.append(NNet(obssize, actsize, sess, optimizer))

sess.run(tf.global_variables_initializer())

env_state_dict = env.reset()
ss = env_state_dict["states"] # assume state is given as an entry in env_state_dict

# lists that store the most recent decision state and performed action of each agent
prev_actions = np.full(nElevator, None)
prev_states = np.full(nElevator, None)

# main iteration of training
# assume env.now() gives the total elapsed simulation time in seconds
while env.now() <= simulation_hours * 3600:
    decision_agents = env_state_dict["decision agents"]
    # time_elapsed_list = env_state_dict["time_elapsed"] # a list of nElevator reals, with all other entries 0, but the decision elevator entries None or time elapsed since this agent's last decision
    R = env_state_dict["rewards"]
    actions = []
    for i in range(len(decision_agents)):
        agent = decision_agents[i]
        # agent updates its NNet
        if prev_actions[agent] != None: # if not the first time being decision agent
            # update corresponding NNet
            time_elapsed = ss[i][-1]
            legal_action_binary = np.zeros(actsize)
            for a in env.legal_actions(agent):
                legal_action_binary[a] = 1
            min_Q_val = np.amin(np.multiply(Q[agent].compute_Qvalues([ss[i]])[0], legal_action_binary))
            target = R[i] + np.exp(-beta * time_elapsed) * min_Q_val
            Q[agent].train([prev_states[agent]], [prev_actions[agent]], [target])

        # agent makes an action choice
        T = 2.0 * factor ** (round(env.now()/3600,4))
        prob_dist = np.zeros(actsize)
        for a in env.legal_actions(agent):
            print("please", Q[agent].compute_Qvalues([ss[i]]))
            q_val = Q[agent].compute_Qvalues([ss[i]])[0][a] # Qi(s,a)
            prob_dist[a] = np.exp(q_val/T)
        prob_dist = prob_dist/sum(prob_dist)
        action = np.random.choice(actsize, p = prob_dist)
        # update
        prev_actions[agent] = action
        prev_states[agent] = ss[i]
        actions.append(action)

    env_state_dict = timed_function(env.step)(actions) # assume env will be modified so that R[i] is in the dict
    # env needs to set R[agent] = 0
    ss = env_state_dict["states"]
    env.render()
