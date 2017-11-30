import tensorflow as tf


class Summary_Parameters():

    def __init__(self):

        self.total_reward = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.episode_length = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.total_loss = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.policy_loss = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.value_loss = tf.Variable(0, dtype=tf.float32, trainable=False)

        tf.summary.scalar('total rewards', self.total_reward)
        tf.summary.scalar('episode length', self.episode_length)
        tf.summary.scalar('total loss', self.total_loss)
        tf.summary.scalar('policy loss', self.policy_loss)
        tf.summary.scalar('value loss', self.value_loss)