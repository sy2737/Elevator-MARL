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
        # build the prediction graph
        states = tf.placeholder(tf.float32, [None, obssize]) # input
        dense1 = tf.layers.dense(state, 20, activation=tf.sigmoid)
        Qvalues = tf.layers.dense(dense1,actsize)

        targets = tf.placeholder(tf.float32, [None])
        actions = tf.placeholder(tf.int32, [None])

        actions_one_hot = tf.one_hot(actions, actsize)

        Qpreds = tf.reduce_sum(tf.multiply(Qvalues, actions_one_hot), axis=1)
        loss = tf.reduce_mean(Qpreds - targets)

        # optimization
        self.train_op = optimizer.minimize(loss)

        # some bookkeeping
        self.Qvalues = Qvalues
        self.states = states
        self.actions = actions
        self.targets = targets
        self.loss = loss
        self.sess = sess

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
        return self.sess.run([self.loss,self.train_op], feed_dict={self.state:states, self.actions:actions, self.targets:targets})
