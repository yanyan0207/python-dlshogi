import subprocess
from argparse import ArgumentParser
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('train_data_list')
    parser.add_argument('test_data_list')
    parser.add_argument('--batch_size', nargs='*', type=int)
    parser.add_argument('--learning_rate', nargs='*', type=float)
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()

    train_py = os.path.join(os.path.dirname(__file__),
                            '../pydlshogi_v2/trainer/trainer_v2_sfen.py')

    for batch_size in args.batch_size:
        for lr in args.learning_rate:
            try:
                model_dir = os.path.join(os.path.dirname(
                    __file__), f'../training_results/trainer_v2-batchsize_{batch_size}-lr_{lr}')
                os.mkdir(model_dir)
                cmd = f'python3 {train_py} {args.train_data_list} {args.test_data_list}'
                cmd += f' --model {model_dir}'
                cmd += f' --batch_size {batch_size}'
                cmd += f' --learning_rate {lr}'
                cmd += f' --epoch {args.epoch}'
                cmd += f' > {model_dir}/result.txt'
                print(cmd)
                subprocess.call(cmd, shell=True)
            except BaseException as e:
                print('error', e)
