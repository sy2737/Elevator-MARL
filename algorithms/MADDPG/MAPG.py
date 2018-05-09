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
import random
logger = gym.logger.get_my_logger(__name__)



class Actor():
    def __init__(self, obssize, actsize, clip_eps, step_per_train, sess, optimizer, global_step):
        """
        obssize: size of the states
        actsize: size of the actions
        """
        # Building the prediction graph
        L = 100
        M = 50
        state = tf.placeholder(tf.float32, [None, obssize])
        W1 = tf.Variable(tf.truncated_normal([obssize, L],stddev=0.01))
        B1 = tf.Variable(tf.truncated_normal([L], stddev=0.01))
        W2 = tf.Variable(tf.truncated_normal([L, M],stddev=0.01))
        B2 = tf.Variable(tf.truncated_normal([M], stddev=0.01))
        W3 = tf.Variable(tf.truncated_normal([M, actsize],stddev=0.01))
        B3 = tf.Variable(tf.truncated_normal([actsize], stddev=0.01))
        
        
        Z1 = tf.sigmoid(tf.matmul(state, W1) + B1)
        Z2 = tf.sigmoid(tf.matmul(Z1, W2) + B2)
        logit = tf.matmul(Z2, W3) + B3
        #logit = tf.matmul(Z1, W2) + B2
        
        # An array of legal action representation (legal=1, illegal=0)
        legal_actions = tf.placeholder(tf.float32, shape=[None,actsize])

        # Manually calculating softmax over selected (legal) indices only
        exp_logit = tf.exp(logit)
        prob = tf.multiply(tf.divide(exp_logit, tf.reduce_sum(tf.multiply(exp_logit, legal_actions))),legal_actions)
        #prob = tf.nn.softmax(tf.multiply(logit, legal_actions))  # prob is of shape [None, actsize]
        
        # BUILD LOSS: Proximal Policy Optimization
        # Advantage estimation from the experiences
        Q_estimate = tf.placeholder(tf.float32, [None])
        # Action probabilities that the previous iteration of network would predict
        old_prob = tf.placeholder(tf.float32, [None, actsize])
        # Action indices that were picked in the experience
        actions = tf.placeholder(tf.int32, [None])
        # Encode those actions into one_hot encoding of length actsize
        actions_one_hot = tf.one_hot(actions, depth=actsize)
        # Select only the relevant probability for new and old probabilities
        prob_i = tf.reduce_sum(tf.multiply(prob, actions_one_hot), axis=1)
        old_prob_i = tf.reduce_sum(tf.multiply(old_prob, actions_one_hot), axis=1)
        
        ratio = tf.divide(prob_i, old_prob_i)

        surrogate_loss = tf.negative(tf.reduce_mean(tf.minimum(
            tf.multiply(ratio, Q_estimate),
            tf.multiply(tf.clip_by_value(ratio, 1-clip_eps, 1+clip_eps), Q_estimate)
        ))) + tf.reduce_sum(prob*tf.log(prob + 1e-9))
        
        
        self.train_op = optimizer.minimize(surrogate_loss, global_step = global_step)
        
        # some bookkeeping
        self.state          = state
        self.prob           = prob
        self.old_prob       = old_prob
        self.actions        = actions
        self.legal_actions  = legal_actions
        self.Q_estimate     = Q_estimate
        self.loss           = surrogate_loss
        self.clip_eps       = clip_eps
        self.step_per_train = step_per_train
        self.optimizer      = optimizer
        self.sess           = sess
    
    def compute_prob(self, states, legal_actions):
        """
        compute prob over actions given states pi(a|s)
        states: numpy array of size [numsamples, obssize]
        legal_actions: array of size [numsamples, actsize], 0 if illegal 1 if legal in that state
        return: numpy array of size [numsamples, actsize]
        """
        
        return self.sess.run(self.prob, feed_dict={self.state:states, self.legal_actions:legal_actions})

    def train(self, states, actions, Qs, old_prob, legal_actions):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        for i in range(self.step_per_train):
            self.sess.run(self.train_op, feed_dict={self.state:states, self.actions:actions, self.Q_estimate:Qs, self.old_prob:old_prob, self.legal_actions:legal_actions})


class Critic():
    def __init__(self, obssize, sess, optimizer, global_step):
        # YOUR CODE HERE
        # need to implement both prediction and loss
        L = 50
        #M = 20
        state = tf.placeholder(tf.float32, [None, obssize])
        W1 = tf.Variable(tf.truncated_normal([obssize, L], stddev=0.01))
        B1 = tf.Variable(tf.truncated_normal([L], stddev=0.01))
        W2 = tf.Variable(tf.truncated_normal([L, 1],stddev=0.01))
        B2 = tf.Variable(tf.truncated_normal([1], stddev=0.01))
        #W3 = tf.Variable(tf.truncated_normal([M, 1],stddev=0.1))
        #B3 = tf.Variable(tf.truncated_normal([1], stddev=0.1))
        
        Z1 = tf.sigmoid(tf.matmul(state, W1) + B1)
        #Z2 = tf.sigmoid(tf.matmul(Z1, W2) + B2)
        val = tf.matmul(Z1, W2) + B2
        
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

def eval_func(actors, actsize, nElevator, env):
    eval_episodes = 15
    record = [] #avg wait time of served passengers from each trajectory
    record2 = [] #avg reward for each elevator in each trajectory
    record3 = [] #avg number of passengers served
    for ite in range(eval_episodes):
        env.spawnRates = spawnRates
        obs = env.reset()
        rsum = {agent: 0 for agent in range(nElevator)}
        while env.now()<3600*1:
            decision_agents  = obs["decision agents"]
            states           = obs["states"]
            rewards          = obs["rewards"]
            #env.render()
            parsed_states = gym.Environment.parse_states(states[0], nFloor, nElevator)

            #logger.info("{:40s}: {}".format("Rewards of decision elevators",rewards))

            #logger.info("{:40s}: {}".format("hall_calls_up", "".join([
            #    "{:8.3}".format(t) for t in parsed_states["hall_call_up_times"]
            #])))

            #logger.info("{:40s}: {}".format("hall_calls_down", "".join([
            #    "{:8.3}".format(t) for t in parsed_states["hall_call_down_times"]
            #])))
            #logger.info("{:40s}: {}".format("Floor requests from within elevator {}".format(
            #    decision_agents[0]), parsed_states['requested_calls']
            #))

            #logger.info("{:40s}: {}".format("Number of passengers served", env.nPassenger_served))
            #logger.info("{:40s}: {}".format("Average waiting time of served",env.avg_wait_time()))
            actions = []
            # take actions
            for idx, agent in enumerate(decision_agents):
                legal_actions = sorted(list(env.legal_actions(agent)))
                boolean_legal_actions = np.zeros(actsize)
                boolean_legal_actions[legal_actions] = 1
                prob   = actors[agent].compute_prob(np.expand_dims(states[idx],0), np.expand_dims(boolean_legal_actions, 0)).flatten()
                act_prob = {env.elevators[0].ACTION_FUNCTION_MAP[act].__name__:prob[act] for idx, act in enumerate(legal_actions)}
                #logger.info("{:40s}: {}".format("Decision agent {} has action probabilities".format(agent), act_prob))
                action = np.random.choice(np.arange(actsize), p=prob, size=1)[0]
                actions.append(action)

            # record
            for idx, agent in enumerate(decision_agents):
                rsum[agent] += rewards[idx]
            #logger.info("Sum of reward for each agent: {}".format([rsum[agent] for agent in range(nElevator)]))
            #time.sleep(0.05)
            #logger.info("=======================================================================")
            newobs = env.step(actions)
            obs = newobs
        #logger.info("Total passengers served: {}".format(env.nPassenger_served))
        record.append(env.avg_wait_time())
        record2.append([rsum[agent] for agent in range(nElevator)])
        record3.append(env.nPassenger_served)
    logger.warning("{:40s}: {}, Average:{}".format("Avg wait time of passegners in each episode", np.round(record), np.mean(record)))
    record2 = np.array(record2)
    logger.warning("{:40s}: {}, Average:{}".format("Average reward of elevators in each episode", np.round(np.mean(record2, axis=1)), np.mean(record2)))
    logger.warning("{:40s}: {}, Average:{}".format("Number of Passengers served in each episode", record3, np.mean(record3)))

if __name__=="__main__":
    logging.disable(logging.DEBUG)

    # Network parameters
    #starter_learning_rate      = 3e-4
    lr_alpha                   = 3e-4 # tf.train.exponential_decay(starter_learning_rate, global_step_alpha, 100, 0.95, staircase=True)
    lr_beta                    = 3e-5 # tf.train.exponential_decay(starter_learning_rate, global_step_beta, 100, 0.95, staircase=True)
    numtrajs                   = 32   # num of trajecories to collect at each iteration 
    episode_length             = 2    # Number of hours to run the episode for
    iterations                 = 1000 # total num of iterations
    gamma                      = 0.99 # discount
    lmbda                      = 0.5    # GAE estimation factor
    clip_eps                   = 0.2
    step_per_train             = 3

    # Environment parameters
    nElevator          = 1
    nFloor             = 4
    spawnRates         = [1/60]+[1/120]*(nFloor-1)
    avgWeight          = 135
    weightLimit        = 1200
    loadTime           = 1


    # Initialize environment
    env                = gym.make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime, 1)
    s                  = env.reset()
    obssize            = env.observation_space_size
    actsize            = env.action_space_size
    logger.warning("Action size: {:5d}, State size: {:5d}".format(actsize, obssize))
    # TODO: God's eye obssize

    ## Initialize Tensorflow/network stuff
    sess               = tf.Session()
    actors             = []
    critics            = []
    for i in range(nElevator):
        global_step_alpha          = tf.Variable(0, trainable=False)
        global_step_beta           = tf.Variable(0, trainable=False)
        optimizer_p                = tf.train.AdamOptimizer(lr_alpha)
        optimizer_v                = tf.train.AdamOptimizer(lr_beta)
        actors.append(Actor(obssize, actsize, clip_eps, step_per_train, sess, optimizer_p, global_step_alpha))
        critics.append(Critic(obssize, sess, optimizer_v, global_step_beta)) 

    # Load saved model
    CHECKPOINT_DIR = "./ckpts/" # specify path to the checkpoint file
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # main iteration
    for ite in range(iterations):    
        if ite%10 == 0:
            logger.warning("Iteration:%d"%(ite))
        if ite%10 == 0:
            saver.save(sess, CHECKPOINT_DIR+"model_{}.ckpt".format(ite))
            logger.warning("MODEL SAVED AT ITERATION {}".format(ite))
        
        OBS  = {agent:[] for agent in range(nElevator)}  # observations
        ACTS = {agent:[] for agent in range(nElevator)}  # actions
        ADS  = {agent:[] for agent in range(nElevator)}  # advantages for actors
        TARGETS  = {agent:[] for agent in range(nElevator)}  # targets for critics
        LEGAL_ACTS= {agent:[] for agent in range(nElevator)}  # legal actions encoded in 0's and 1's

        for num in range(numtrajs):
            # record for each episode
            obss = {agent:[] for agent in range(nElevator)}  # observations
            acts = {agent:[] for agent in range(nElevator)}  # actions
            rews = {agent:[] for agent in range(nElevator)}  # instant rewards for one trajectory
            legal_acts = {agent:[] for agent in range(nElevator)} # legal actions encoded in 0's and 1's

            env.spawnRates = spawnRates
            obs = env.reset()

            epi_step = 0
            # Randomly step through the environment for some steps before officially starting to train
            random_start = np.random.randint(50,1000)
            seen_psngr = False
            while env.now()<3600*episode_length:
                epi_step += 1
                decision_agents  = obs["decision agents"]
                states           = obs["states"]
                rewards          = obs["rewards"]
                actions          = []
                b_legal_actions  = [] # Boolean legal actions for one decision epoch! from all agents
                # take actions
                if epi_step <= random_start:
                    actions = [random.sample(env.legal_actions(agent), 1)[0] for agent in decision_agents]
                    newobs = env.step(actions)
                    obs = newobs
                    continue
                #if epi_step == random_start+1:
                #    print("randomly started at step", epi_step)
                #    env.render()
                env.spawnRates = np.zeros(env.nFloor)
                
                for idx, agent in enumerate(decision_agents):
                    # Obtain legal actinos and encode them into 1's and 0's
                    legal_actions                        = sorted(list(env.legal_actions(agent)))
                    boolean_legal_actions                = np.zeros(actsize)
                    boolean_legal_actions[legal_actions] = 1
                    # Probability over all actions (but illegal ones will have probability of zero)
                    prob   = actors[agent].compute_prob(np.expand_dims(states[idx],0), np.expand_dims(boolean_legal_actions, 0)).flatten()
                    action = np.random.choice(np.arange(actsize), p=prob)
                    actions.append(action)
                    b_legal_actions.append(boolean_legal_actions)

                # record
                for idx, agent in enumerate(decision_agents):
                    obss[agent].append(states[idx])
                    acts[agent].append(actions[idx])
                    rews[agent].append(rewards[idx]) # Need to add negation reward if the environment returns cost not reward
                    legal_acts[agent].append(b_legal_actions[idx])
                
                # update
                if env.no_passenger():
                    if seen_psngr:
                        #print("Took {} to reach empty state".format(epi_step-random_start))
                        #env.render()
                        break
                else:
                    seen_psngr = True

                newobs = env.step(actions)
                obs = newobs

            # logger.warning("Episode lasted: {:5d} step!".format(epi_step))

            # Discard the first reward and the last actions
            # because each round the reward we observe correspond to the previous episode
            # We keep the last state because it serves as the next_state to the second to last state
            for agent in range(nElevator):
                acts[agent] = acts[agent][:-1]
                legal_acts[agent] = legal_acts[agent][:-1]
                rews[agent] = rews[agent][1: ]
                
            # Compute discount sum of rewards from instant rewards
            returns         = {agent: discounted_rewards(rews[agent], gamma) for agent in rews}
            #print("returns of one traject:", returns)
            # Compute the GAE
            vals            = {agent: critics[agent].compute_values(ob) for agent, ob in obss.items()}
            bellman_errors  = {agent: np.array(rews[agent]) + (gamma*val[1:] - val[:-1]).reshape(-1) for agent, val in vals.items()}
            GAE             = {agent: discounted_rewards(errors, gamma*lmbda) for agent, errors in bellman_errors.items()}
            # The last val is the value estimation of last state, which we discard
            #GAE             = {agent: list(np.array(returns[agent]) - v.flatten()[:-1]) for agent,v in vals.items()}
            #print("Advantage estimation:", GAE)
            
            # Record for batch update
            for agent in range(nElevator):
                TARGETS[agent] += returns[agent]
                OBS[agent]     += obss[agent][:-1]
                ACTS[agent]    += acts[agent]
                ADS[agent]     += GAE[agent]
                LEGAL_ACTS[agent] += legal_acts[agent]
        
        for agent in range(nElevator):
            # update baseline
            TARGETS[agent]  = np.array(TARGETS[agent])
            OBS[agent]      = np.array(OBS[agent])
            ACTS[agent]     = np.array(ACTS[agent])
            ADS[agent]      = np.array(ADS[agent])
            critics[agent].train(OBS[agent], np.reshape(TARGETS[agent], [-1,1]))
        
            # update policy
            legal_actions = np.array(LEGAL_ACTS[agent])
            old_prob      = actors[agent].compute_prob(OBS[agent], legal_actions)
            actors[agent].train(OBS[agent], ACTS[agent], ADS[agent], old_prob, legal_actions)  # update

        if ite%20 == 0:
            eval_func(actors, actsize, nElevator, env)


