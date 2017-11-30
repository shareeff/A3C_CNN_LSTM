import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as nn
import tensorflow.contrib.slim as slim


SMALL_VALUE = 1e-20

class A3C_Network(object):

    def __init__(self, args, no_action, scope):
        self.scope = scope
        self.lstm_input_dim = args.lstm_input_dim
        self.lstm_size = args.lstm_size
        self.no_action = no_action
        self.initializer = tf.truncated_normal_initializer(stddev=0.02)
        self.biases_initializer = tf.constant_initializer(0.0)
        self.create_network()
        self.checkpoint_path = args.checkpoint_dir
        self.environment = args.environment

    def create_network(self):

        with tf.variable_scope(self.scope):
            self.s = tf.placeholder("float", [None, 84, 84, 4])

            self.conv1 = nn.conv2d(inputs=self.s, num_outputs=16, kernel_size=8, stride=4, \
                                                    padding='valid', activation_fn=tf.nn.relu, \
                                                    biases_initializer=self.biases_initializer, scope='conv1')
            self.conv2 = nn.conv2d(inputs=self.conv1, num_outputs=32, kernel_size=4, stride=2, \
                                   padding='valid', activation_fn=tf.nn.relu, \
                                   biases_initializer=self.biases_initializer, scope='conv2')
            #self.conv3 = nn.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=3, stride=2, \
            #                       padding='SAME', activation_fn=tf.nn.relu, \
            #                      weights_initializer=self.initializer, scope='conv3')
            #self.flatten1 = tf.reshape(self.conv2, shape=[-1, 6400])
            self.flatten1 = slim.flatten(self.conv2)
            self.fc1 = tf.contrib.layers.fully_connected(inputs=self.flatten1, num_outputs=self.lstm_input_dim, \
                                                      activation_fn=tf.nn.relu, \
                                                      biases_initializer = self.biases_initializer, scope='fc1')
            with tf.variable_scope("lstm1"):

                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
                c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
                h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
                self.lstm_state_init = [c_init, h_init]
                c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
                h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
                self.lstm_state_in = (c_in, h_in)
                rnn_in = tf.expand_dims(self.fc1, [0])
                step_size = tf.shape(self.s)[:1]
                state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
                lstm_outputs, lstm_state_out = tf.nn.dynamic_rnn(
                        lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                        time_major=False)
                self.rnn_out = tf.reshape(lstm_outputs, [-1, self.lstm_size])

                lstm_c, lstm_h = lstm_state_out
                self.lstm_state = [lstm_c[:1, :], lstm_h[:1, :]]

            self.policy = tf.contrib.layers.fully_connected(inputs=self.rnn_out, num_outputs=self.no_action, \
                                                             activation_fn=tf.nn.softmax,
                                                             weights_initializer=self.normalized_columns_initializer(0.01),
                                                             scope='policy')   # initializer std 0.01
            self.value = tf.contrib.layers.fully_connected(inputs=self.rnn_out, num_outputs=1, \
                                                            activation_fn=None,
                                                            weights_initializer=self.normalized_columns_initializer(1.0),
                                                            scope='value')  #initializer std 1.0

            self.prepare_loss()

    def prepare_loss(self):
        self.a = tf.placeholder(shape=[None], dtype=tf.int32)
        self.a_onehot = tf.one_hot(self.a, self.no_action, dtype=tf.float32)
        self.y = tf.placeholder(shape=[None], dtype=tf.float32)
        self.advantages =  tf.placeholder(shape=[None], dtype=tf.float32)
        log_policy = tf.log(tf.clip_by_value(self.policy, SMALL_VALUE, 1.0))
        self.readout_action = tf.reduce_sum(tf.multiply(log_policy, self.a_onehot), reduction_indices=1)
        self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.y - tf.reshape(self.value, [-1])))
        self.policy_loss = -tf.reduce_sum(self.readout_action*self.advantages)
        self.entropy = -tf.reduce_sum(self.policy * log_policy)
        self.loss = 0.5 * self.value_loss + self.policy_loss - 0.01 * self.entropy

        grads = tf.gradients(self.loss, self.get_var_list())
        self.var_norms = tf.global_norm(self.get_var_list())
        self.grads, self.grad_norms = tf.clip_by_global_norm(grads, 40.0)



    def get_var_list(self):
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        return self.vars

    def update_graph(self, from_net):

        with tf.variable_scope(self.scope):
            to_vars = self.get_var_list()
            from_vars = from_net.get_var_list()
            op_holder = []
            for from_var, self_var in zip(from_vars,to_vars):
                op_holder.append(self_var.assign(from_var))

            return tf.group(*op_holder)

    def load_model(self, sess, saver):
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('.............Model restored to global.............')
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            print('................No model is found.................')

    def save_model(self, sess, saver, time_step):
        print('............save model ............')
        saver.save(sess, self.checkpoint_path + '/'+self.environment +'-' + str(time_step) + '.ckpt')

    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer


