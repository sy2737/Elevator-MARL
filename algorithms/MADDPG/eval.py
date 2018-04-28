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



class Actor():
    def __init__(self, obssize, actsize, clip_eps, step_per_train, sess, optimizer, global_step):
        """
        obssize: size of the states
        actsize: size of the actions
        """
        # YOUR CODE HERE
        # BUILD PREDICTION GRAPH
        # build the input
        L = 50
        M = 20
        state = tf.placeholder(tf.float32, [None, obssize])
        W1 = tf.Variable(tf.truncated_normal([obssize, L],stddev=0.1))
        B1 = tf.Variable(tf.truncated_normal([L], stddev=0.1))
        W2 = tf.Variable(tf.truncated_normal([L, M],stddev=0.1))
        B2 = tf.Variable(tf.truncated_normal([M], stddev=0.1))
        W3 = tf.Variable(tf.truncated_normal([M, actsize],stddev=0.1))
        B3 = tf.Variable(tf.truncated_normal([actsize], stddev=0.1))
        
        
        Z1 = tf.nn.relu(tf.matmul(state, W1) + B1)
        Z2 = tf.nn.relu(tf.matmul(Z1, W2) + B2)
        logit = tf.matmul(Z2, W3) + B3
        logit = tf.Print(logit,[logit], "actor logit: ")
        
        prob = tf.nn.softmax(logit)  # prob is of shape [None, actsize]
        
        # BUILD LOSS
        Q_estimate = tf.placeholder(tf.float32, [None])
        old_prob = tf.placeholder(tf.float32, [None, actsize])
        actions = tf.placeholder(tf.int32, [None])
        actions_onehot = tf.one_hot(actions, depth=actsize)
        

        prob_i = tf.reduce_sum(tf.multiply(prob, actions_onehot), axis=1)
        old_prob_i = tf.reduce_sum(tf.multiply(old_prob, actions_onehot), axis=1)
        
        ratio = tf.divide(prob_i, old_prob_i)
        
        surrogate_loss = tf.negative(tf.reduce_mean(tf.minimum(
            tf.multiply(ratio, Q_estimate),
            tf.multiply(tf.clip_by_value(ratio, 1-clip_eps, 1+clip_eps), Q_estimate)
        )))
        
        
        self.train_op = optimizer.minimize(surrogate_loss, global_step = global_step)
        
        # some bookkeeping
        self.state = state
        self.prob = prob
        self.old_prob = old_prob
        self.actions = actions
        self.Q_estimate = Q_estimate
        self.loss = surrogate_loss
        self.clip_eps = clip_eps
        self.step_per_train = step_per_train
        self.optimizer = optimizer
        self.sess = sess
    
    def compute_prob(self, states):
        """
        compute prob over actions given states pi(a|s)
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples, actsize]
        """
        # YOUR CODE HERE
        
        return self.sess.run(self.prob, feed_dict={self.state:states})

    def train(self, states, actions, Qs, old_prob):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        for i in range(self.step_per_train):
            self.sess.run(self.train_op, feed_dict={self.state:states, self.actions:actions, self.Q_estimate:Qs, self.old_prob:old_prob})


class Critic():
    def __init__(self, obssize, sess, optimizer, global_step):
        # YOUR CODE HERE
        # need to implement both prediction and loss
        L = 50
        M = 20
        state = tf.placeholder(tf.float32, [None, obssize])
        W1 = tf.Variable(tf.truncated_normal([obssize, L], stddev=0.1))
        B1 = tf.Variable(tf.truncated_normal([L], stddev=0.1))
        W2 = tf.Variable(tf.truncated_normal([L, M],stddev=0.1))
        B2 = tf.Variable(tf.truncated_normal([M], stddev=0.1))
        W3 = tf.Variable(tf.truncated_normal([M, 1],stddev=0.1))
        B3 = tf.Variable(tf.truncated_normal([1], stddev=0.1))
        
        Z1 = tf.nn.relu(tf.matmul(state, W1) + B1)
        Z2 = tf.nn.relu(tf.matmul(Z1, W2) + B2)
        val = tf.matmul(Z2, W3) + B3
        
        target = tf.placeholder(tf.float32, [None, 1])
        loss = tf.losses.mean_squared_error(target, val)
        self.train_op = optimizer.minimize(loss, global_step = global_step)
        
        self.state = state
        self.val = val
        self.target = target
        self.loss = loss
        self.sess = sess
        self.optimizer = optimizer

    def compute_values(self, states):
        """
        compute value function for given states
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples]
        """
        # YOUR CODE HERE
        return self.sess.run(self.val, feed_dict={self.state:states})

    def train(self, states, targets):
        """
        states: numpy array
        targets: numpy array
        """
        # YOUR CODE HERE
        return self.sess.run(self.train_op, feed_dict={self.state:states, self.target:targets})


def discounted_rewards(r, lmbda):
    """ take 1D float array of rewards and compute discounted bellman errors """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * lmbda + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)

logger = gym.logger.get_my_logger(__name__)
if __name__=="__main__":
    logging.disable(logging.INFO)

    # Network parameters
    starter_learning_rate      = 1e-3
    lr_alpha                   = 3e-4 # tf.train.exponential_decay(starter_learning_rate, global_step_alpha, 100, 0.95, staircase=True)
    lr_beta                    = 3e-4 # tf.train.exponential_decay(starter_learning_rate, global_step_beta, 100, 0.95, staircase=True)
    numtrajs                   = 5    # num of trajecories to collect at each iteration 
    episode_length             = 5    # Number of hours to run the episode for
    iterations                 = 50   # total num of iterations
    gamma                      = 1    # discount
    lmbda                      = 1    # GAE estimation factor
    clip_eps                   = 0.2
    step_per_train             = 10

    # Environment parameters
    nElevator          = 1
    nFloor             = 3
    spawnRates         = [1/60]+[1/180]*(nFloor-1)
    avgWeight          = 135
    weightLimit        = 1200
    loadTime           = 1

    env = gym.make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime)
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

    eval_mode = True

    obs = env.reset()
    done = False
    rsum = {agent: 0 for agent in range(nElevator)}
    while env.now()<3600*episode_length:
        decision_agents  = obs["decision agents"]
        states           = obs["states"]
        rewards          = obs["rewards"]
        env.render()
        parsed_states = gym.Environment.parse_states(states[0], nFloor, nElevator)

        logger.info("Rewards of decision elevators: {}".format(rewards))

        logger.info("  hall_calls_up: \n"+"".join([
            "{:8.3}".format(t) for t in parsed_states["hall_call_up_times"]
        ]))

        logger.info("  hall_calls_down: \n"+"".join([
            "{:8.3}".format(t) for t in parsed_states["hall_call_down_times"]
        ]))
        logger.info("Floor requests from within elevator {}:\n{}".format(
            decision_agents[0], parsed_states['requested_calls']
        ))

        logger.info("Number of passengers served:{}".format(env.nPassenger_served))
        logger.info("Average waiting time of served passengers: {}".format(env.avg_wait_time()))
        actions = []
        # take actions
        for idx, agent in enumerate(decision_agents):
            legal_actions = list(env.legal_actions(agent))
            prob   = actors[agent].compute_prob(np.expand_dims(states[idx],0)).flatten()
            print("COmputed probability vector", prob)
            prob   = np.array([prob[i] for i in legal_actions])
            if sum(prob)==0:
                prob = np.ones(len(legal_actions))
            prob   = prob/sum(prob)
            act_prob = {act:prob[idx] for idx, act in enumerate(legal_actions)}
            print("Decision agent {} has action probabilities {}".format(agent, act_prob))

            action = np.random.choice(legal_actions, p=prob, size=1)
            actions.append(action[0])

        # record
        for idx, agent in enumerate(decision_agents):
            rsum[agent] += rewards[idx]
        logger.info("Sum of reward for each agent: {}".format([rsum[agent] for agent in range(nElevator)]))
        time.sleep(0.5)
        newobs = env.step(actions)
        obs = newobs