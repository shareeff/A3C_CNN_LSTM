import tensorflow as tf
import numpy as np
#import scipy.signal
import cv2
from A3C_Network import A3C_Network
from Summary import *

class Worker(object):

    def __init__(self, global_episodes,training_episodes, master_net, id, learning_rate, env, summary_writer,
                 summary_parameters, write_op, args):

        self.name = 'worker_' + str(id)
        self.global_episodes = global_episodes
        self.increse_global_episodes = global_episodes.assign_add(1)
        self.training_episodes = training_episodes
        self.increase_training_episodes = training_episodes.assign_add(1)
        self.summary_writer = summary_writer
        self.summary_parameters = summary_parameters
        self.writer_op = write_op
        self.gamma = args.gamma
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=args.decay, epsilon=args.epsilon)
        self.master_net = master_net
        self.env = env
        self.no_action = master_net.no_action
        self.worker_net = A3C_Network(args, self.no_action, self.name)
        self.update_local_net = self.worker_net.update_graph(master_net)
        self.apply_grads = self.trainer.apply_gradients(list(zip(self.worker_net.grads, self.master_net.get_var_list())))
        self.T = 0
        self.No_Training = 0
        self.batch_size = args.batch_size


    def train(self, sess, bootstrap_value):


        reward_batch = np.array(self.reward_batch)
        value_batch = np.array(self.value_batch)
        np.clip(reward_batch, -1.0, 1.0, out=reward_batch)

        R_batch = self.discount(reward_batch, bootstrap_value)

        A_batch = self.calculate_advantage(reward_batch, value_batch, bootstrap_value)

        training_episodes = sess.run(self.increase_training_episodes)
        self.value_loss, self.policy_loss, self.total_loss, _ = \
            sess.run([self.worker_net.value_loss, self.worker_net.policy_loss, self.worker_net.loss, self.apply_grads],
                                           feed_dict = {
                                               self.worker_net.s : self.observation_batch,
                                               self.worker_net.a : self.action_batch,
                                               self.worker_net.y : R_batch,
                                               self.worker_net.advantages : A_batch,
                                               self.worker_net.lstm_state_in[0] : self.lstm_state_train[0],
                                               self.worker_net.lstm_state_in[1] : self.lstm_state_train[1]

                                           })

        sess.run(self.update_local_net)

        self.observation_batch = []
        self.action_batch = []
        self.reward_batch = []
        self.value_batch = []

        self.No_Training +=1

        return training_episodes



    def process(self, sess, coord, saver):

        terminal = True
        a_indexes = np.arange(self.no_action)
        self.observation_batch = []
        self.action_batch = []
        self.reward_batch = []
        self.value_batch = []
        training_episodes = 0


        while not coord.should_stop():

            if terminal:
                terminal = False
                self.lstm_state = self.worker_net.lstm_state_init
                x_t = self.env.reset()
                x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
                current_observation = np.stack((x_t, x_t, x_t, x_t), axis=2)
                self.total_reward = 0
                episode_length = 0

            self.lstm_state_train = self.lstm_state

            for _ in range(0, self.batch_size):

                global_episodes = sess.run(self.increse_global_episodes)
                a_dist, value, self.lstm_state = sess.run([self.worker_net.policy, self.worker_net.value,
                                                              self.worker_net.lstm_state], feed_dict={
                                                        self.worker_net.s : [current_observation],
                                                        self.worker_net.lstm_state_in[0] : self.lstm_state[0],
                                                        self.worker_net.lstm_state_in[1] : self.lstm_state[1]

                                                      })

                a = np.random.choice(a_indexes, p=a_dist[0])
               # a_t = np.argmax(a_dist == a)
                x_t1, r_t, terminal, info = self.env.step(a)
                self.total_reward += r_t
                episode_length += 1

                x_t1 = cv2.cvtColor(cv2.resize(x_t1, (84, 84)), cv2.COLOR_BGR2GRAY)
                next_observation = np.stack((x_t1, x_t1, x_t1, x_t1), axis=2)

                self.observation_batch.append(current_observation)
                self.action_batch.append(a)
                self.reward_batch.append(r_t)
                self.value_batch.append(value[0, 0])
                self.T += 1
                current_observation = next_observation

                if terminal:
                    print('ID :' + self.name + ', global episode :' + str(
                        global_episodes) + ', global training step :' + str(training_episodes) +', local training step :'
                          + str(self.No_Training) +  ', total reward :' + str(self.total_reward))
                    break


            if not terminal:
                bootstrap_value = sess.run(self.worker_net.value, feed_dict={
                                                        self.worker_net.s : [current_observation],
                                                        self.worker_net.lstm_state_in[0] : self.lstm_state[0],
                                                        self.worker_net.lstm_state_in[1] : self.lstm_state[1]})[0,0]
            else:
                bootstrap_value = 0.0

            training_episodes = self.train(sess, bootstrap_value)

            if terminal:
                summary = sess.run(self.writer_op,
                                   {self.summary_parameters.total_reward: float(self.total_reward),
                                    self.summary_parameters.episode_length: float(episode_length),
                                    self.summary_parameters.total_loss: float(self.total_loss),
                                    self.summary_parameters.value_loss: float(self.value_loss),
                                    self.summary_parameters.policy_loss: float(self.policy_loss)})

                self.summary_writer.add_summary(summary, global_episodes)
                self.summary_writer.flush()


            if global_episodes % 5000 == 0:

                self.master_net.save_model(sess, saver, global_episodes)

    def discount(self, r, bootstrap):
        size = len(r)
        R_batch = np.zeros([size], np.float64)
        R = bootstrap
        for i in reversed(range(0, size)):
            R = r[i] + self.gamma * R
            R_batch[i] = R
        return R_batch

    def calculate_advantage(self, r, v, bootstrap):
        size = len(r)
        A_batch = np.zeros([size], np.float64)
        aux = bootstrap
        for i in reversed(range(0, size)):
            aux = r[i] + self.gamma * aux
            A = aux - v[i]
            A_batch[i] = A
        return A_batch




