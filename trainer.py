import tensorflow as tf
import gym
from A3C_Network import A3C_Network
from Worker import Worker
from config import *
from Summary import *
import threading
from time import sleep



try:
    tf.reset_default_graph()
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    training_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

    #summary_writer = tf.summary.FileWriter(args.summary_dir)


    n_threads = args.num_thread
    Env = gym.make(args.environment)
    no_action = Env.action_space.n
    Env.close()
    learning_rate = tf.train.polynomial_decay(args.learning_rate, global_episodes, args.decay_steps,
                                              args.learning_rate * 0.1)
    #trainer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate, decay=args.decay)

    summary_writer = []
    for id in range(n_threads):
        summary_writer.append(tf.summary.FileWriter(args.summary_dir+'/worker_'+str(id)))

    summary_parameters = Summary_Parameters()
    write_op = tf.summary.merge_all()
    master_network = A3C_Network(args, no_action, 'master_network')
    workers = []
    env_list = []
    for id in range(n_threads):
        env = gym.make(args.environment)

        if id == 0:
            env = gym.wrappers.Monitor(env, "monitors", force=True)

        workers.append(Worker(global_episodes, training_episodes, master_network, id, learning_rate, env, summary_writer[id],
                              summary_parameters, write_op, args))
        env_list.append(env)


    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)
        master_network.load_model(sess, saver)
        coord = tf.train.Coordinator()
        thread_list = []
        for id in range(n_threads):
            t = threading.Thread(target=workers[id].process, args=(sess, coord, saver))
            t.start()
            sleep(0.5)
            thread_list.append(t)

        coord.join(thread_list)
        for t in thread_list:
            t.start()

        print("Ctrl + C to close")
        coord.wait_for_stop()


except KeyboardInterrupt:

    print("Closing threads")
    coord.request_stop()

    print("Closing environments")
    for env in env_list:
        env.close()

    sess.close()


