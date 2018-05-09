# Implementing the MADDPG paper, but using stochastic policy gradient instead of Deterministic PG
# since the environment has discrete action space.
# When the central critic is conditioned on actions of all 
import sys
sys.path.insert(0, '/Users/stevenydc/Google Drive/Semester 2/elevator_rl')
sys.path.insert(0, 'C:\\Users\\steve\\Google Drive\\Semester 2\\elevator_rl\\')
import environment as gym
import tensorflow as tf
import numpy as np
import logging
import time
import os
from MAPG import Actor, Critic, discounted_rewards


logger = gym.logger.get_my_logger(__name__)
if __name__=="__main__":
    logging.disable(logging.DEBUG)

    # Network parameters
    lr_alpha                   = 3e-4 # tf.train.exponential_decay(starter_learning_rate, global_step_alpha, 100, 0.95, staircase=True)
    lr_beta                    = 3e-4 # tf.train.exponential_decay(starter_learning_rate, global_step_beta, 100, 0.95, staircase=True)
    numtrajs                   = 5    # num of trajecories to collect at each iteration 
    episode_length             = 1    # Number of hours to run the episode for
    iterations                 = 50   # total num of iterations
    gamma                      = 1    # discount
    lmbda                      = 1    # GAE estimation factor
    clip_eps                   = 0.2
    step_per_train             = 10

    # Environment parameters
    nElevator          = 1
    nFloor             = 4
    spawnRates         = [1/60]+[1/120]*(nFloor-1)
    avgWeight          = 135
    weightLimit        = 1200
    loadTime           = 1

    env = gym.make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime, 0)
    obssize            = env.observation_space_size
    actsize            = env.action_space_size
    logger.warning("Action size: {:5d}, State size: {:5d}".format(actsize, obssize))
    ## Initialize Tensorflow/network stuff
    sess               = tf.Session()
    actors = []
    critics = []
    for i in range(nElevator):
        global_step_alpha          = tf.Variable(0, trainable=False)
        global_step_beta           = tf.Variable(0, trainable=False)
        optimizer_p                = tf.train.AdamOptimizer(lr_alpha)
        optimizer_v                = tf.train.AdamOptimizer(lr_beta)
        actors.append(Actor(obssize, actsize, clip_eps, step_per_train, sess, optimizer_p, global_step_alpha))
        critics.append(Critic(obssize, sess, optimizer_v, global_step_beta)) 

    # Load saved model
    CHECKPOINT_DIR = "./ckpts/" # specify path to the checkpoint file
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    print("Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)


    obs = env.reset()
    done = False
    rsum = {agent: 0 for agent in range(nElevator)}
    while env.now()<3600*episode_length:
        decision_agents  = obs["decision agents"]
        states           = obs["states"]
        rewards          = obs["rewards"]
        env.render()
        parsed_states = gym.Environment.parse_states(states[0], nFloor, nElevator)

        logger.info("{:40s}: {}".format("Rewards of decision elevators",rewards))

        logger.info("{:40s}: {}".format("hall_calls_up", "".join([
            "{:8.3}".format(t) for t in parsed_states["hall_calls_up"]
        ])))

        logger.info("{:40s}: {}".format("hall_calls_down", "".join([
            "{:8.3}".format(t) for t in parsed_states["hall_calls_down"]
        ])))
        logger.info("{:40s}: {}".format("Floor requests from within elevator {}".format(
            decision_agents[0]), parsed_states['requested_calls']
        ))

        logger.info("{:40s}: {}".format("Number of passengers served", env.nPassenger_served))
        logger.info("{:40s}: {}".format("Average waiting time of served",env.avg_wait_time()))
        actions = []
        # take actions
        for idx, agent in enumerate(decision_agents):
            legal_actions = sorted(list(env.legal_actions(agent)))
            boolean_legal_actions = np.zeros(actsize)
            boolean_legal_actions[legal_actions] = 1
            prob   = actors[agent].compute_prob(np.expand_dims(states[idx],0), np.expand_dims(boolean_legal_actions, 0)).flatten()
            act_prob = {env.elevators[0].ACTION_FUNCTION_MAP[act].__name__:prob[act] for idx, act in enumerate(legal_actions)}
            logger.info("{:40s}: {}".format("Decision agent {} has action probabilities".format(agent), act_prob))
            action = np.random.choice(np.arange(actsize), p=prob, size=1)[0]
            actions.append(action)

        # record
        for idx, agent in enumerate(decision_agents):
            rsum[agent] += rewards[idx]
        logger.info("Sum of reward for each agent: {}".format([rsum[agent] for agent in range(nElevator)]))
        time.sleep(0.5)
        logger.info("=======================================================================")
        newobs = env.step(actions)
        obs = newobs
    logger.info("Total passengers served: {}".format(env.nPassenger_served))