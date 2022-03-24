import os
import numpy as np
from absl import flags, app
from model import MLQP
from utils import dataloader, set_seed
from tqdm import trange
import pandas as pd

FLAGS = flags.FLAGS
# flags.DEFINE_enum('mode', 'train', ['train', 'test'], "Execution mode")
flags.DEFINE_float('lr', 1e-2, "Learning rate")
flags.DEFINE_integer('num_hidden', 32, "Number of hidden neurons")
flags.DEFINE_integer('num_epoch', 2000, "Number of epoches")
flags.DEFINE_integer('seed', 0, "Random seed")
flags.DEFINE_string('name', "mlqp", "Experiment name")

def train():
    train_data = dataloader("train")
    test_data = dataloader("test")

    mlqp = MLQP(1, 2, FLAGS.num_hidden, 1)

    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    with trange(1, FLAGS.num_epoch + 1, desc='Training', ncols=0) as pbar:
        for epoch in pbar:
            #train
            preds = []
            for cor, y in train_data:
                pred = mlqp.forward(cor)
                mlqp.backward(y)
                mlqp.step(FLAGS.lr)
                preds.append(pred)
            preds = np.squeeze(np.array(preds))
            loss = 0.5 * (y - preds) ** 2
            acc = ((preds > 0.5) == train_data[2]).mean()
            pbar.set_postfix(loss='%.4f' % loss, acc='%.4f' % acc)
            train_loss.append(loss)
            train_acc.append(acc)

            # test
            if epoch % 50 == 0:
                preds = []
                for cor, y in zip(test_data):
                    pred = mlqp.forward(cor)
                    preds.append(pred)
                preds = np.squeeze(np.array(preds))
                loss = 0.5 * (y - preds) ** 2
                acc = ((preds > 0.5) == train_data[2]).mean()
                pbar.set_postfix(loss='%.4f' % loss, acc='%.4f' % acc)
                test_loss.append(loss)
                test_acc.append(acc)

    log_dir = os.path.join("logs", FLAGS.name)
    os.makedirs(os.path.join(log_dir, "visualize"))
    dataframe = pd.DataFrame({'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': train_loss, 'test_acc': train_acc})
    dataframe.to_csv(os.path.join(log_dir, "log.csv"), sep=',')


def main(argv):
    set_seed(FLAGS.seed)
    train()


if __name__ == '__main__':
    app.run(main)