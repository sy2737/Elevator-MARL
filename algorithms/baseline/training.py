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
if len(sys.argv) <= 1:
    print("Input Format: python training.py nElevator(int) nFloor(int) lr(float) simulation_hours(int) Q-learning-factor(float) outpulFilePath(str) inputFilePath(str) debug_mode(bool)")
    exit(1)
print(" nElevator = ", sys.argv[1], "\n",
      "nFloor = ", sys.argv[2], "\n",
      "Learning Rate = ", sys.argv[3], "\n",
      "Simulation Hourse = ", sys.argv[4], "\n",
      "Q-Learning Factor = ", sys.argv[5], "\n",
      "Output File Path = ", sys.argv[6], "\n",
      "Input File Path = ", sys.argv[7] if len(sys.argv) == 9 else "None", "\n",
      "Debug Mode = ", sys.argv[8] if len(sys.argv) == 9 else sys.argv[7], "\n")

answer = input('Starting training?: [y/n]')
if not answer or answer[0].lower() != 'y':
    exit(1)

"""
Prep work
"""
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
    # user defined parameters
    nElevator = int(sys.argv[1])
    nFloor = int(sys.argv[2])
    lr = float(sys.argv[3]) # NNet learning rate
    simulation_hours = int(sys.argv[4]) # number of simulation hours to run
    factor = float(sys.argv[5]) # specific for Q-learning-ext, controls exploration behavior
    debug = sys.argv[8] if len(sys.argv) == 9 else sys.argv[7]
    # other parameters
    spawnRates = [1/360]+[1/360]*(nFloor-1)
    avgWeight = 135
    weightLimit = 1200
    loadTime = 1
    beta = 0.01 # exponential decay factor for target computation



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
if len(sys.argv) == 9:
    input_ckpt_path = sys.argv[7] # specify path to the checkpoint file
    saver = tf.train.Saver()
    saver.restore(sess, input_ckpt_path)
    print("Model restored.")
else: # if train from scratch
    sess.run(tf.global_variables_initializer())

# specify path to store the trained model
output_ckpt_path = sys.argv[6]

# create a list to store avg waiting time after every 10 simulation hours
wait_time_list = []

"""
Begin training
"""
env_state_dict = env.reset()
ss = env_state_dict["states"] # a list of states, one state for each decision agent

# lists that store the most recent decision state and performed action of each agent
prev_actions = np.full(nElevator, None)
prev_states = np.full(nElevator, None)

# main iteration of training
# env.now() gives the total elapsed simulation time in seconds
last_checkpoint_time = 0
print_counter = 0
while env.now() <= simulation_hours * 3600:
    # save updated graph after every 10 hours of training
    # remember temperature matters; so cannot train in segments
    if env.now() - last_checkpoint_time > 10 * 3600:
        tf.train.Saver().save(sess, output_ckpt_path)
        print("graph saved after ", round(env.now()/3600, 1), "simulation hours")
        last_checkpoint_time = env.now() # set last_checkpoint_time to current time

    decision_agents = env_state_dict["decision agents"]
    R = env_state_dict["rewards"]
    actions = []
    print_counter += 1
    # each decision agent updates NNet and selects action
    for i in range(len(decision_agents)):
        agent = decision_agents[i]

        # construct legal_actions_list, in ascending order of action index
        legal_actions_list = list(env.legal_actions(agent))
        legal_actions_list.sort()
        # construct a boolean representation of legal_actions
        legal_actions_bool = np.full(actsize, False)
        for action in env.legal_actions(agent):
            legal_actions_bool[action] = True
        # compute q-values
        q_vals = Q[agent].compute_Qvalues([ss[i]])[0]
        loss = None

        # Update NNet
        if prev_actions[agent] != None: # if not the first time being a decision agent
            # update corresponding NNet
            time_elapsed = ss[i][-1]
            # compute min Q value in current state
            min_Q_val = np.amin(q_vals[legal_actions_list])
            # compute target Q value of PREVIOUS state, action pair
            target = R[i] + np.exp(-beta * time_elapsed) * min_Q_val
            # update agent's NNet parameters
            loss, _ = Q[agent].train([prev_states[agent]], [prev_actions[agent]], [target])

        # Choose action
        # compute temperature used in softmax action distribution
        T = 2.0 * (factor ** (round(env.now()/3600,4)))
        # compute a probability distribution on legal actions
        # len(prob_dist) = len(legal_actions)
        # in ascending order of legal actions' indices
        prob_dist, _ = Q[agent].compute_legal_action_prob_dist([ss[i]], legal_actions_bool, T)

        # sample a legal action from prob_dist of legal actions
        action = np.random.choice(legal_actions_list, p = prob_dist)

        # update prev_actions and prev_states lists
        prev_actions[agent] = action
        prev_states[agent] = ss[i]
        actions.append(action)

    # if debug mode is on, enable visualiztion
    if debug:
        if print_counter % 500 in [0, 1, 2, 3, 4, 5]:
            if print_counter % 500 == 0:
                print("----------------------------------------")
                print(print_counter, "->", print_counter + 5)
                print("----------------------------------------")
            env.render()

            print("temperature: ", T)
            for i in range(len(decision_agents)):
                print("decision agent ", decision_agents[i])
                print("action to take: ", actions[i])
                print("requested calls from within: ", ss[i][-1-nFloor:-1])
            if env.nPassenger_served >= 1 and print_counter % 500 ==5:
                print("avg wait time is: ", env.avg_wait_time())
                wait_time_list.append(env.avg_wait_time())


    # take actions
    env_state_dict = timed_function(env.step)(actions)
    ss = env_state_dict["states"]



# save the final model
tf.train.Saver().save(sess, output_ckpt_path)
print("avg waiting time every 10 simulation_hours = ", wait_time_list)
