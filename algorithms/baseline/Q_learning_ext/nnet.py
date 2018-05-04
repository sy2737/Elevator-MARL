import tensorflow as tf
import numpy as np

# define neural net Q_\theta(s,a) as a class

class NNet(object):

    def __init__(self, obssize, actsize, sess, optimizer):
        """
        obssize: dimension of state space
        actsize: dimension of action space
        sess: sess to execute this Qfunction
        optimizer:
        """
        # placeholders
        legal_actions = tf.placeholder(tf.bool, actsize)
        temperature = tf.placeholder(tf.float32)
        targets = tf.placeholder(tf.float32, [None])
        actions = tf.placeholder(tf.int32, [None])

        # build the prediction graph
        L = np.int(np.floor(actsize/2))
        M = np.int(np.floor(L/2))

        states = tf.placeholder(tf.float32, [None, obssize])
        W1 = tf.Variable(tf.truncated_normal([obssize, L],stddev=0.1))
        B1 = tf.Variable(tf.truncated_normal([L], stddev=0.1))

        W2 = tf.Variable(tf.truncated_normal([L, M],stddev=0.1))
        B2 = tf.Variable(tf.truncated_normal([M], stddev=0.1))

        W3 = tf.Variable(tf.truncated_normal([M, actsize],stddev=0.1))
        B3 = tf.Variable(tf.truncated_normal([actsize], stddev=0.1))

        Z1 = tf.nn.relu(tf.matmul(states, W1) + B1)
        Z2 = tf.nn.relu(tf.matmul(Z1, W2) + B2)

        Qvalues = tf.matmul(Z2, W3)+B3

        # compute quantities of interest
        actions_one_hot = tf.one_hot(actions, actsize)
        Qpreds = tf.reduce_sum(tf.multiply(Qvalues, actions_one_hot), axis=1)
        loss = tf.reduce_mean(tf.square(Qpreds - targets))

        # compute softmax prob distribution on legal actions
        legal_action_Qvals = tf.boolean_mask(Qvalues[0], legal_actions)
        legal_actions_dist = tf.nn.softmax(-legal_action_Qvals/temperature)

        # optimization
        self.train_op = optimizer.minimize(loss)

        # some bookkeeping
        self.Qvalues = Qvalues
        self.states = states
        self.actions = actions
        self.targets = targets
        self.loss = loss
        self.sess = sess
        self.legal_actions = legal_actions
        self.temperature = temperature
        self.legal_actions_dist = legal_actions_dist
        self.legal_action_Qvals = legal_action_Qvals

    def compute_Qvalues(self, states):
        """
        states: numpy array as input to the neural net, states should have
        size [numsamples, obssize], where numsamples is the number of samples
        output: Q values for these states. The output should have size
        [numsamples, actsize] as numpy array
        """
        return self.sess.run(self.Qvalues, feed_dict={self.states: states})

    def train(self, states, actions, targets):
        """
        states: numpy array as input to compute loss (s)
        actions: numpy array as input to compute loss (a)
        targets: numpy array as input to compute loss (Q targets)
        """
        return self.sess.run([self.loss, self.train_op], feed_dict={self.states:states, self.actions:actions, self.targets:targets})

    def compute_legal_action_prob_dist(self, states, legal_actions, temperature):
        """
        legal_actions: numpy array as input to prob_dist
        """
        return self.sess.run([self.legal_actions_dist, self.legal_action_Qvals], feed_dict={self.states:states, self.legal_actions:legal_actions, self.temperature:temperature})
