import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--environment', type=str, default='Breakout-v0')
parser.add_argument('--checkpoint_dir', type=str, default='./save_model')
parser.add_argument('--summary_dir', type=str, default='./summary_log')
parser.add_argument('--learning_rate', type=float, default=7e-4)  # 0.00001
parser.add_argument('--decay', type=float, default=0.99)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--decay_steps', type=int, default=250000000)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--lstm_size', type=int, default=256)
parser.add_argument('--lstm_input_dim', type=int, default=256)
parser.add_argument('--num_thread', type=int, default=4)   #4
parser.add_argument('--gamma', type=float, default= 0.99)


args = parser.parse_args()