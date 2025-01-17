import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('model_root', nargs='*')
    args = parser.parse_args()

    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot']
    for i, model_root in enumerate(args.model_root):
        if os.path.exists(f'{model_root}/history.csv'):
            l, c = divmod(i, len(color_list))
            df = pd.read_csv(f'{model_root}/history.csv')
            # plt.plot(df.policy_output_loss, label=os.path.basename(
            #    model_root), color=color_list[c], linestyle=linestyle_list[l])
            plt.plot(df.val_policy_output_loss, label=os.path.basename(
                model_root), color=color_list[c], linestyle=linestyle_list[l])
    plt.legend()
    plt.figure()
    for i, model_root in enumerate(args.model_root):
        l, c = divmod(i, len(color_list))
        if os.path.exists(f'{model_root}/history.csv'):
            df = pd.read_csv(f'{model_root}/history.csv')
            plt.plot(df.val_policy_output_accuracy, label=os.path.basename(
                model_root), color=color_list[c], linestyle=linestyle_list[l])
    plt.legend()
    plt.figure()
    for i, model_root in enumerate(args.model_root):
        l, c = divmod(i, len(color_list))
        if os.path.exists(f'{model_root}/model_match.csv'):
            df = pd.read_csv(f'{model_root}/model_match.csv')
            plt.plot(df.win_rate, label=os.path.basename(
                model_root), color=color_list[c], linestyle=linestyle_list[l])
    plt.legend()
    plt.show()
