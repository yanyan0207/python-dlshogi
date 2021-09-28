from argparse import ArgumentParser
from pydlshogi_v2.trainer.trainer_v2 import createDataSet
from pydlshogi_v2.features.features_v2 import FeaturesV2
from pydlshogi_v2.features.position_list import readPositionListCsv
from pydlshogi_v2.models import resnet_v2
import sys

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow_addons.optimizers import RectifiedAdam

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('train_position_list_csv')
    parser.add_argument('test_position_list_csv')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--train_min_rate', type=int, default=2500)
    parser.add_argument('--train_max_num', type=int)
    parser.add_argument('--test_min_rate', type=int, default=2500)
    parser.add_argument('--test_max_num', type=int)
    parser.add_argument('--min_move_num', type=int, default=50)
    parser.add_argument('--model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--moment', type=float, default=0.0)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--total_steps', type=int, default=0)

    args = parser.parse_args()
    df_train = readPositionListCsv(
        args.train_position_list_csv, min_move_num=args.min_move_num, min_rate=args.train_min_rate, max_num=args.train_max_num)
    df_test = readPositionListCsv(
        args.test_position_list_csv, min_move_num=args.min_move_num, min_rate=args.test_min_rate, max_num=args.test_max_num)

    print(df_train.shape)
    print(df_test.shape)

    features = FeaturesV2()

    train_ds = createDataSet(df_train, args.batch_size,
                             features, shuffle=False)
    test_ds = createDataSet(df_test, 4096, features, shuffle=False)

    model = resnet_v2.createModel()

    if args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.learning_rate,
                        momentum=args.moment, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = Adam(learning_rate=args.learning_rate)
    elif args.optimizer == 'radam':
        optimizer = RectifiedAdam(
            learning_rate=args.learning_rate, total_steps=args.total_steps)
    else:
        raise('unknown optimzer:' + args.optimizer)

    loss_policy_fn = SparseCategoricalCrossentropy(from_logits=False)

    # エポックごと
    for epoch in range(args.epoch):
        # ミニバッチ
        for step, (x_batch_train, (y_policy_batch_train, y_value_batch_train)) in enumerate(train_ds):
            # 勾配の記録
            with tf.GradientTape() as tape:
                logits_policy, logits_value = model(x_batch_train)

                loss_policy_value = loss_policy_fn(
                    y_policy_batch_train, logits_policy)

            grads = tape.gradient(
                loss_policy_value, model.trainable_weights)[0]

            model.summary()

            layer = model.layers[3]
            layer_3_weights = layer.get_weights()
            weights = layer.get_weights()[0]
            # plt.figure()
            #plt.hist(weights.reshape(-1), bins=1024)
            # plt.figure()
            #plt.hist(grads[0].numpy().reshape(-1), bins=1024)
            # plt.yscale('log')

            print("loss_policy_value", loss_policy_value)
            max_idx = np.argmax(grads.numpy())
            max_grad = grads.numpy().reshape(-1)[max_idx]
            weights_value = weights.reshape(-1)[max_idx]
            print(max_idx, max_grad)

            def updateWeights0(idx, value):
                tmp = layer_3_weights[0].reshape(-1)
                tmp[idx] = value
                layer.set_weights(
                    [tmp.reshape(layer_3_weights[0].shape), layer_3_weights[1]])

            def calc_loss_policy():
                logits_policy, logits_value = model(
                    x_batch_train, training=False)

                loss_policy_value = loss_policy_fn(
                    y_policy_batch_train, logits_policy)

                return loss_policy_value

            weights_list = []
            loss_list = []
            for i in range(1000):
                weights_value -= max_grad / 1000
                updateWeights0(max_idx, weights_value)
                weights_list.append(weights_value)
                loss_list.append(calc_loss_policy())
                print(weights_value, loss_list[-1])
            plt.plot(weights_list, loss_list)
            plt.show()
            sys.exit()
