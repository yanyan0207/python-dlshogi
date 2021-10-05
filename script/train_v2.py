import subprocess
from argparse import ArgumentParser
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', nargs='*', type=int)
    parser.add_argument('--learning_rate', nargs='*', type=float)
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()

    train_py = os.path.join(os.path.dirname(__file__),
                            'pydlshogi_v2/trainer/trainer_v2_sfen.py')
    train_data_list = os.path.join(
        os.environ['HOME'], 'data/floodgate/data_list_2017_minrate_2500_minmove_50.csv')
    test_data_list = os.path.join(
        os.environ['HOME'], 'data/floodgate/data_list_2020_minrate_4000_minmove_50.csv')

    for batch_size in args.batch_size:
        for lr in args.learning_rate:
            try:
                model_dir = os.path.join(os.path.dirname(
                    __file__), f'training_results/trainer_v2-batchsize_{batch_size}-lr_{lr}')
                os.mkdir(model_dir)
                cmd = f'python {train_py} {train_data_list} {test_data_list}'
                cmd += f' --model {model_dir}'
                cmd += f' --batch_size {batch_size}'
                cmd += f' --learning_rate {lr}'
                cmd += f' --epoch {args.epoch}'
                cmd += f' > {model_dir}/result.txt'
                print(cmd)
                subprocess.call(cmd, shell=True)
            except BaseException as e:
                print('error', e)
