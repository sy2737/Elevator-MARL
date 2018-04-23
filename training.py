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
    logging.disable(logging.INFO)
    nElevator = 5
    nFloor = 10
    spawnRates = [1/30]+[1/180]*(nFloor-1)
    avgWeight = 135
    weightLimit = 1200
    loadTime = 1
    lr = 1e-4
    beta = 0.01

# initialize environment
env = gym.make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime)
obssize = env.observation_space_size
actsize = env.action_space_size
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

# Use the saver to restore trained model from disk
saver = tf.train.Saver()
saver.restore(sess, "./q_learning_ext_model.ckpt")
print("Model restored.")

## if train from scratch, used the code below
# sess.run(tf.global_variables_initializer())

env_state_dict = env.reset()
ss = env_state_dict["states"]

# lists that store the most recent decision state and performed action of each agent
prev_actions = np.full(nElevator, None)
prev_states = np.full(nElevator, None)

# main iteration of training
# env.now() gives the total elapsed simulation time in seconds
last_checkpoint_time = 0
while env.now() <= simulation_hours * 3600:
    print("current time is: -----", env.now(), "-----")
    # save updated graph after every 10 hours of training.
    # remember temperature matters, so that cannot train in segments
    if env.now() - last_checkpoint_time > 10 * 3600:
        tf.train.Saver().save(sess, "./q_learning_ext_model.ckpt")
        print("graph saved after ", round(env.now()/3600, 1), "simulation hours")
        last_checkpoint_time = env.now()
    decision_agents = env_state_dict["decision agents"]
    R = env_state_dict["rewards"]
    actions = []
    for i in range(len(decision_agents)):
        agent = decision_agents[i]
        q_vals = Q[agent].compute_Qvalues([ss[i]])[0]
        # agent updates its NNet
        if prev_actions[agent] != None: # if not the first time being decision agent
            # update corresponding NNet
            time_elapsed = ss[i][-1]
            legal_action_binary = np.zeros(actsize)
            for a in env.legal_actions(agent):
                legal_action_binary[a] = 1
            min_Q_val = np.amin(np.multiply(q_vals, legal_action_binary))
            target = R[i] + np.exp(-beta * time_elapsed) * min_Q_val
            # print("reward of agent ", agent, " is: ", R[i])
            print("min Q-val is: ", min_Q_val)
            print("target is: ", target, "; current Q-val is: ", Q[agent].compute_Qvalues([prev_states[agent]])[0][prev_actions[agent]])
            Q[agent].train([prev_states[agent]], [prev_actions[agent]], [target])

        # agent makes an action choice
        T = 2.0 * factor ** (round(env.now()/3600,4))
        
        # nLegal_actions = len(env.legal_actions(agent))
        # prob_dist = np.zeros(nLegal_actions)
        # for i in range(nLegal_actions):
        #     a = env.legal_actions(agent)[i]
        #     q_val = q_vals[a] # Qi(s,a)
        #     prob_dist[i] = np.exp(q_val/T)
        # prob_dist = prob_dist/sum(prob_dist)
        action = np.random.choice(env.legal_actions(agent), p = prob_dist)
        # update prev_actions and prev_states lists
        prev_actions[agent] = action
        prev_states[agent] = ss[i]
        actions.append(action)

    env_state_dict = timed_function(env.step)(actions)
    ss = env_state_dict["states"]

    ## enable visualization
    # env.render()

tf.train.Saver().save(sess, "./q_learning_ext_model.ckpt")
