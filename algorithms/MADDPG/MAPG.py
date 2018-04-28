# Implementing the MADDPG paper, but using stochastic policy gradient instead of Deterministic PG
# since the environment has discrete action space.
# When the central critic is conditioned on actions of all 
import sys
#sys.path.insert(0, '/Users/stevenydc/Google Drive/Semester 2/elevator_rl')
sys.path.insert(0, 'C:\\Users\\steve\\Google Drive\\Semester 2\\elevator_rl\\')
import environment as gym
import tensorflow as tf
import numpy as np



class Actor():
    def __init__(self, obssize, actsize, clip_eps, step_per_train, sess, optimizer, global_step):
        """
        obssize: size of the states
        actsize: size of the actions
        """
        # YOUR CODE HERE
        # BUILD PREDICTION GRAPH
        # build the input
        L = 30
        M = 15
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
        L = 30
        M = 15
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

if __name__=="__main__":
    logger = gym.logger.get_my_logger(__name__)

    # Network parameters
    starter_learning_rate      = 1e-3
    lr_alpha                   = 3e-4 # tf.train.exponential_decay(starter_learning_rate, global_step_alpha, 100, 0.95, staircase=True)
    lr_beta                    = 3e-4 # tf.train.exponential_decay(starter_learning_rate, global_step_beta, 100, 0.95, staircase=True)
    numtrajs                   = 10   # num of trajecories to collect at each iteration 
    episode_length             = 12   # Number of hours to run the episode for
    iterations                 = 1000 # total num of iterations
    gamma                      = .99  # discount
    lmbda                      = 1    # GAE estimation factor
    clip_eps                   = 0.2
    step_per_train             = 10

    # Environment parameters
    nElevator          = 3
    nFloor             = 14
    spawnRates         = [1/60]+[1/180]*(nFloor-1)
    avgWeight          = 135
    weightLimit        = 1200
    loadTime           = 1

    def eval_func():
        eval_episodes = 100
        record = []
        env = gym.make()
        eval_mode = True
        for ite in range(eval_episodes):
            obs = env.reset()
            done = False
            rsum = 0
            while env.now()<3600*episode_length:
                decision_agents  = obs["decision agents"]
                states           = obs["states"]
                rewards          = obs["rewards"]
                actions = []
                # take actions
                for idx, agent in enumerate(decision_agents):
                    prob   = actors[agent].compute_prob(np.expand_dims(states[idx],0))
                    action = np.random.choice(actsize, p=prob.flatten(), size=1)
                    actions.append(action[0])

                # record
                for idx, agent in enumerate(decision_agents):
                    obss[agent].append(states[idx])
                    acts[agent].append(actions[idx])
                    rews[agent].append(rewards[idx])
                
                newobs = env.step(actions)

                # update
                obs = newobs

            record.append(env.avg_wait_time())

        logger.info("Average wait time of passegners: {}".format(np.mean(record)))

    # Initialize environment
    env                = gym.make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime)
    s                  = env.reset()
    obssize            = env.observation_space_size
    actsize            = env.action_space
    # TOOD: God's eye obssize

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

    sess.run(tf.global_variables_initializer())


    # main iteration
    for ite in range(iterations):    
        if ite%10 == 0:
            logger.info("Iteration:%d"%(ite))
        
        OBS  = {agent:[] for agent in range(nElevator)}  # observations
        ACTS = {agent:[] for agent in range(nElevator)}  # actions
        ADS  = {agent:[] for agent in range(nElevator)}  # advantages for actors
        TARGETS  = {agent:[] for agent in range(nElevator)}  # targets for critics

        for num in range(numtrajs):
            # record for each episode
            obss = {agent:[] for agent in range(nElevator)}  # observations
            acts = {agent:[] for agent in range(nElevator)}  # actions
            rews = {agent:[] for agent in range(nElevator)}  # instant rewards for the previous episode!!

            obs = env.reset()

            done = False
            epi_step = 0
            while env.now()<3600*episode_length:
                epi_step += 1
                decision_agents  = obs["decision agents"]
                states           = obs["states"]
                rewards          = obs["rewards"]
                actions = []
                # take actions
                for idx, agent in enumerate(decision_agents):
                    prob   = actors[agent].compute_prob(np.expand_dims(states[idx],0))
                    action = np.random.choice(actsize, p=prob.flatten(), size=1)
                    actions.append(action[0])

                # record
                for idx, agent in enumerate(decision_agents):
                    obss[agent].append(states[idx])
                    acts[agent].append(actions[idx])
                    rews[agent].append(rewards[idx])
                
                newobs = env.step(actions)

                # update
                obs = newobs
            # discard the first reward and the last actions
            # because each round the reward we observe correspond to the previous episode
            # We keep the last state because it serves as the next_state to the second to last state
            for agent in range(nElevator):
                acts[agent] = acts[agent][:-1]
                rews[agent] = rews[agent][1: ]
                
            # compute discount sum of rewards from instant rewards
            returns         = {agent: discounted_rewards(rews[agent], gamma) for agent in rews}
            # compute the gaes 
            vals            = {agent: critics[agent].compute_values(obs) for agent, obs in obss.items()}
            bellman_errors  = {agent: np.array(rews[agent]) + (gamma*val[1:] - val[:-1]).reshape(-1) for agent, val in vals.items()}
            GAE             = {agent: discounted_rewards(errors, gamma*lmbda) for agent, errors in bellman_errors.items()}
            
            # record for batch update
            for agent in range(nElevator):
                TARGETS[agent] += returns[agent]
                OBS[agent] += obss[agent][:-1]
                ACTS[agent] += acts[agent]
                ADS[agent] += GAE[agent]
        
        # update baseline
        for agent in range(nElevator):
            TARGETS[agent]  = np.array(TARGETS[agent])
            OBS[agent]      = np.array(OBS[agent])
            ACTS[agent]     = np.array(ACTS[agent])
            ADS[agent]      = np.array(ADS[agent])
        
            critics[agent].train(OBS, np.reshape(TARGETS[agent], [-1,1]))
        
            # update policy
            old_prob = actors[agent].compute_prob(OBS[agent])
            actors[agent].train(OBS[agent], ACTS[agent], ADS[agent], old_prob)  # update
        if ite%50 == 0:
            eval_func()


