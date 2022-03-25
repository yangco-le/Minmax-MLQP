import os
import time
import numpy as np
from absl import flags, app
from model import MLQP
from utils import *
from tqdm import trange
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'minmax', ['vanilla', 'minmax'], "Execution mode")
flags.DEFINE_float('lr', 1e-1, "Learning rate")
flags.DEFINE_integer('num_hidden', 32, "Number of hidden neurons")
flags.DEFINE_integer('num_epoch', 10000, "Number of epoches")
flags.DEFINE_integer('seed', 0, "Random seed")
flags.DEFINE_enum('partition', "radius", ["random", "yaxis", "radius"], "Partition mode")
flags.DEFINE_string('name', "minmax_lr1e-1", "Experiment name")

def train():
    train_data = dataloader("train")
    test_data = dataloader("test")

    mlqp = MLQP(1, 2, FLAGS.num_hidden, 1)

    log_dir = os.path.join("logs", FLAGS.name)
    os.makedirs(os.path.join(log_dir, "visualize"), exist_ok=True)

    time0 = time.time()
    r_time = 0
    train_loss, train_acc, test_loss, test_acc, li_time = [], [], [], [], []
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
            loss = (0.5 * (train_data[:, 1] - preds) ** 2).sum()
            acc = ((preds > 0.5) == train_data[:, 1]).mean()
            pbar.set_postfix(loss='%.4f' % loss, acc='%.4f' % acc)

            # test
            if epoch % 10 == 0:
                # training log 
                train_loss.append(loss)
                train_acc.append(acc)

                # testing log
                preds = []
                for cor, y in test_data:
                    pred = mlqp.forward(cor)
                    preds.append(pred)
                preds = np.squeeze(np.array(preds))
                loss_test = (0.5 * (test_data[:, 1] - preds) ** 2).sum()
                acc_test = ((preds > 0.5) == test_data[:, 1]).mean()
                pbar.set_postfix(loss='%.4f' % loss, acc='%.4f' % acc)
                test_loss.append(loss_test)
                test_acc.append(acc_test)

                # time log
                li_time.append(time.time() - time0 - r_time)
            
            if epoch % 1000 == 0:
                r_time0 = time.time()
                # plot decision boundary
                plot_decision_boundary(lambda x: mlqp.forward(x), np.stack(train_data[:, 0]).T, train_data[:, 1], 
                                       os.path.join(log_dir, "visualize", "epoch_{}".format(epoch)))
                r_time += time.time() - r_time0

    dataframe = pd.DataFrame({'time': li_time, 'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 
                              'test_acc': test_acc})
    dataframe.to_csv(os.path.join(log_dir, "log.csv"), sep=',')

def train_minmax():
    train_data = dataloader("train")
    one_set, zero_set = partition_data(dataloader("train"), FLAGS.partition)
    train_data_tmp = []
    for i in range(2):
        for j in range(2):
            train_data_tmp.append(one_set[i] + zero_set[j])
    test_data = dataloader("test")

    models = []
    for i in range(4):
        models.append(MLQP(1, 2, FLAGS.num_hidden, 1))

    log_dir = os.path.join("logs", FLAGS.name)
    os.makedirs(os.path.join(log_dir, "visualize"), exist_ok=True)

    time0 = time.time()
    r_time = 0
    train_loss, train_acc, test_loss, test_acc, li_time = [], [], [], [], []
    with trange(1, FLAGS.num_epoch + 1, desc='Training', ncols=0) as pbar:
        for epoch in pbar:
            # train
            for i in range(2):
                for j in range(2):
                    tmp_data = train_data_tmp[2 * i + j]
                    for cor, y in tmp_data:
                        pred = models[2 * i + j].forward(cor)
                        models[2 * i + j].backward(y)
                        models[2 * i + j].step(FLAGS.lr)
            
            preds = [[], [], [], []]
            for i in range(4):
                for x, y in train_data:
                    preds[i].append(models[i].forward(x)[0][0])
            preds = np.array(preds)
            preds = np.max((np.min((preds[0], preds[1]), 0), np.min((preds[2], preds[3]), 0)), 0)
            loss = (0.5 * (train_data[:, 1] - preds) ** 2).sum()
            acc = ((preds > 0.5) == train_data[:, 1]).mean()
            pbar.set_postfix(loss='%.4f' % loss, acc='%.4f' % acc)

            # test
            if epoch % 10 == 0:
                # training log 
                train_loss.append(loss)
                train_acc.append(acc)

                # testing log
                preds = [[], [], [], []]
                for i in range(4):
                    for x, y in test_data:
                        preds[i].append(models[i].forward(x)[0][0])
                preds = np.array(preds)
                preds = np.max((np.min((preds[0], preds[1]), 0), np.min((preds[2], preds[3]), 0)), 0)
                loss_test = (0.5 * (test_data[:, 1] - preds) ** 2).sum()
                acc_test = ((preds > 0.5) == test_data[:, 1]).mean()
                pbar.set_postfix(loss='%.4f' % loss, acc='%.4f' % acc)
                test_loss.append(loss_test)
                test_acc.append(acc_test)

                # time log
                li_time.append(time.time() - time0 - r_time)
            
            if epoch % 1000 == 0:
                r_time0 = time.time()
                # plot decision boundary
                plot_decision_boundary(lambda x: models[0].forward(x), 
                                       np.stack(np.array(train_data_tmp[0])[:, 0]).T, 
                                       np.array(train_data_tmp[0])[:, 1], 
                                       os.path.join(log_dir, "visualize", "epoch_{}_model_1".format(epoch)))
                plot_decision_boundary(lambda x: models[1].forward(x), 
                                       np.stack(np.array(train_data_tmp[1])[:, 0]).T, 
                                       np.array(train_data_tmp[1])[:, 1], 
                                       os.path.join(log_dir, "visualize", "epoch_{}_model_2".format(epoch)))
                plot_decision_boundary(lambda x: models[2].forward(x), 
                                       np.stack(np.array(train_data_tmp[2])[:, 0]).T, 
                                       np.array(train_data_tmp[2])[:, 1], 
                                       os.path.join(log_dir, "visualize", "epoch_{}_model_3".format(epoch)))
                plot_decision_boundary(lambda x: models[3].forward(x), 
                                       np.stack(np.array(train_data_tmp[3])[:, 0]).T,
                                       np.array(train_data_tmp[3])[:, 1], 
                                       os.path.join(log_dir, "visualize", "epoch_{}_model_4".format(epoch)))
                plot_decision_boundary(lambda x: min(models[0].forward(x), models[1].forward(x)), 
                                       np.stack(np.array(train_data_tmp[0] + train_data_tmp[1])[:, 0]).T, 
                                       np.array(train_data_tmp[0] + train_data_tmp[1])[:, 1], 
                                       os.path.join(log_dir, "visualize", "epoch_{}_model_12".format(epoch)))
                plot_decision_boundary(lambda x: min(models[2].forward(x), models[3].forward(x)), 
                                       np.stack(np.array(train_data_tmp[2] + train_data_tmp[3])[:, 0]).T, 
                                       np.array(train_data_tmp[2] + train_data_tmp[3])[:, 1], 
                                       os.path.join(log_dir, "visualize", "epoch_{}_model_34".format(epoch)))
                plot_decision_boundary(lambda x: max(min(models[0].forward(x), models[1].forward(x)), min(models[2].forward(x), models[3].forward(x))), 
                                       np.stack(train_data[:, 0]).T, 
                                       train_data[:, 1], 
                                       os.path.join(log_dir, "visualize", "epoch_{}_model_1234".format(epoch)))
                r_time += time.time() - r_time0

    dataframe = pd.DataFrame({'time': li_time, 'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 
                              'test_acc': test_acc})
    dataframe.to_csv(os.path.join(log_dir, "log.csv"), sep=',')


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.mode == "vanilla":
        train()
    elif FLAGS.mode == "minmax":
        train_minmax()


if __name__ == '__main__':
    app.run(main)