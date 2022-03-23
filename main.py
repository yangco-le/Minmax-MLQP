import os
import numpy
from absl import flags, app
from model import MLQP
from utils import set_seed

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'test'], "Execution mode")
flags.DEFINE_float('lr', 2e-4, "Learning rate")
flags.DEFINE_integer('num_hidden', 32, "Number of hidden neurons")
flags.DEFINE_integer('epoch', 20000, "Number of epoches")
flags.DEFINE_integer('seed', 0, "Random seed")

def train():
    pass

def test():
    pass

def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.mode == "train":
        train()
    else:
        test()

if __name__ == '__main__':
    app.run(main)