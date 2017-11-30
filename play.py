import tensorflow as tf
import gym
import cv2
import numpy as np
from A3C_Network import A3C_Network
from config import *

NUM_OF_GAMES = 10

env = gym.make(args.environment).unwrapped
no_action = env.action_space.n
x_t = env.reset()
x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
game_state = np.stack((x_t, x_t, x_t, x_t), axis=2)



play_network = A3C_Network(args, no_action, 'master_network')

sess = tf.InteractiveSession()

saver = tf.train.Saver(max_to_keep=5)
play_network.load_model(sess, saver)

lstm_state = play_network.lstm_state_init
score = 0

game_no = 1
while True:

    env.render()

    a, lstm_state = sess.run([play_network.policy, play_network.lstm_state], feed_dict={
                                                       play_network.s : [game_state],
                                                       play_network.lstm_state_in[0] : lstm_state[0],
                                                        play_network.lstm_state_in[1] : lstm_state[1]

                                                      })

    action = np.argmax(a)
    x_t1, r_t, terminal, info = env.step(action)

    x_t1 = cv2.cvtColor(cv2.resize(x_t1, (84, 84)), cv2.COLOR_BGR2GRAY)

    game_state = np.stack((x_t1, x_t1, x_t1, x_t1), axis=2)
    score += r_t

    if terminal:
        lstm_state = play_network.lstm_state_init
        print('Game No : '+ str(game_no)+ ', Score : ', score)
        x_t = env.reset()
        x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
        game_state = np.stack((x_t, x_t, x_t, x_t), axis=2)
        game_no += 1
        score = 0
        if game_no == NUM_OF_GAMES:
            break




