FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

WORKDIR /root

ENV DEBIAN_FRONTEND noninteractive

# 必要パッケージのインストール
RUN apt-get update  \
    && apt-get -y upgrade \
    && apt-get install -y \
    git \
    python3 \
    python3-pip \
    && apt-get autoremove -y \
    && apt-get clean  \
    && rm -r /var/lib/apt/lists/*

# pythonパッケージのインストール
RUN pip3 install -U pip \
    && pip install --no-cache-dir \
    tensorflow-gpu \
    tensorflow_addons \
    scipy \
    pandas \
    matplotlib \
    joblib \
    python-shogi

# 棋譜ファイルの追加
ADD floodgate/wdoor2017.tar.gz /root/data/floodgate/
ADD floodgate/wdoor2020.tar.gz /root/data/floodgate/

# python dlshogiのダウンロード
RUN git clone https://github.com/yanyan0207/python-dlshogi.git
WORKDIR /root/python-dlshogi
RUN git fetch && git checkout v2
RUN pip3 install -e .

# 前処理
WORKDIR /root/data/floodgate
RUN python3 ~/python-dlshogi/pydlshogi_v2/features/kifu_list.py wdoor2017 kifulist_2017.csv
RUN python3 ~/python-dlshogi/pydlshogi_v2/features/sfen_list.py --min_rate 2500 --min_move_num 50 \
    kifulist_2017.csv sfenlist_2017_min-rate_2500_min-move-num_50.csv 
RUN python3 ~/python-dlshogi/pydlshogi_v2/features/data_list.py \
    sfenlist_2017_min-rate_2500_min-move-num_50.csv  datalist_2017_min-rate_2500_min-move-num_50.csv 

RUN python3 ~/python-dlshogi/pydlshogi_v2/features/kifu_list.py wdoor2020 kifulist_2020.csv
RUN python3 ~/python-dlshogi/pydlshogi_v2/features/sfen_list.py --min_rate 4000 --min_move_num 50 \
    kifulist_2020.csv sfenlist_2020_min-rate_4000_min-move-num_50.csv 
RUN python3 ~/python-dlshogi/pydlshogi_v2/features/data_list.py \
    sfenlist_2020_min-rate_4000_min-move-num_50.csv  datalist_2020_min-rate_4000_min-move-num_50.csv 

# 訓練
WORKDIR /root/python-dlshogi/training_results
RUN python3 ../script/train_v2.py --batch_size 128 64 32  --learning_rate 0.08 0.04 0.02 --epoch 4 \
    ~/data/floodgate/datalist_2017_min-rate_2500_min-move-num_50.csv \
    ~/data/floodgate/datalist_2020_min-rate_4000_min-move-num_50.csv 

# 自己対戦
RUN python3 ../selfmatch/model_match.py trainer_v2-batchsize_*
