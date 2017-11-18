import tensorflow as tf
import numpy as np
from collections import deque
from netutil import *
import random

np.random.seed(1)
tf.set_random_seed(1)
INPUT_LENGTH_SIZE = 80
INPUT_WIDTH_SIZE = 80
INPUT_CHANNEL = 4
ACTIONS_DIM = 2

NN_UNITS = 512
MEMORY_SIZE = 50000


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=50000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize memory
        self.memory = deque(maxlen=memory_size)

        self.memory_counter = 0

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.initialize_all_variables())

        self.cost_his = []
        self.saver = tf.train.Saver()

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('lstm'):
                lstm_in = tf.concat([s[:, :, :, 0], s[:, :, :, 1], s[:, :, :, 2], s[:, :, :, 3]], 1)
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(2, forget_bias=1.0, state_is_tuple=True)
                outputs, states = tf.nn.dynamic_rnn(lstm_cell, lstm_in,  dtype=tf.float32, time_major=False)
                outputs = outputs[:, 3, :]

            with tf.variable_scope('l4'):
                w_f1 = tf.get_variable('w_f1', [2, NN_UNITS], initializer=w_initializer,
                                       collections=c_names)
                b_f1 = tf.get_variable('b_f1', [NN_UNITS], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.matmul(outputs, w_f1) + b_f1)

            with tf.variable_scope('l5'):
                w_a_f2 = tf.get_variable('w_a_f2', [NN_UNITS, ACTIONS_DIM], initializer=w_initializer,
                                         collections=c_names)
                b_a_f2 = tf.get_variable('b_a_f2', [ACTIONS_DIM], initializer=b_initializer, collections=c_names)
                self.A = tf.matmul(l4, w_a_f2) + b_a_f2
                self.A = tf.nn.softmax(self.A)

            with tf.variable_scope('Q'):
                out = self.A
            return out
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder('float', shape=[None, 1, 2, INPUT_CHANNEL])  # input State
        self.label = tf.placeholder('float', shape=[None, 2])

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.truncated_normal_initializer(mean=0.0, stddev=0.01), tf.constant_initializer(0.01)  # config of layers
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval, self.label, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

    def store_transition(self, s, label):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        self.memory.append((s, label))
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        max_Q = np.max(actions_value)
        # else:
        #     action = np.random.randint(0, self.n_actions)
        return action, max_Q

    def learn(self, temp_lr):
        # check to replace target parameters
        self.lr = temp_lr
        batch = random.sample(self.memory, self.batch_size)
        state_batch = [t[0] for t in batch]
        label_batch = [t[1] for t in batch]
        my_loss = self.sess.run(self._train_op, feed_dict={self.s: state_batch, self.label: label_batch})
        if self.learn_step_counter % 5000 == 0:
            self.saver.save(self.sess, 'record/save_net.ckpt')
        self.cost_his.append(my_loss)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return my_loss

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(2, output_graph=True)



