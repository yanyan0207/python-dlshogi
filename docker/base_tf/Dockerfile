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
    pandas \
    joblib \
    python-shogi

# 棋譜ファイルの追加
ADD floodgate/wdoor2017.tar.gz /~/data/floodgate/
ADD floodgate/wdoor2020.tar.gz /~/data/floodgate/

# python dlshogiのダウンロード
RUN git clone https://github.com/yanyan0207/python-dlshogi.git
WORKDIR /root/python-dlshogi
RUN git checkout v0.0.2
RUN pip3 install -e .
WORKDIR /root/

# 前処理
RUN python3 python-dlshogi/utils/create_kifulist.py --kifu_root /~/data/floodgate /~/data/floodgate/wdoor2017 kifulist_2017.csv
RUN python3 python-dlshogi/utils/create_positionlist.py --kifu_root /~/data/floodgate --min_rate 2500 --min_move_num 50 \
    kifulist_2017.csv positionlist_2017_min-rate_2500_min-move-num_50.csv 

RUN python3 python-dlshogi/utils/create_kifulist.py --kifu_root /~/data/floodgate /~/data/floodgate/wdoor2020 kifulist_2020.csv
RUN python3 python-dlshogi/utils/create_positionlist.py --kifu_root /~/data/floodgate --min_rate 4000 --min_move_num 50 \
    kifulist_2020.csv positionlist_2020_min-rate_4000_min-move-num_50.csv 

# policy
RUN mkdir model_2017_policy
RUN nvidia-smi
RUN python3 python-dlshogi/script/tf/train_policy.py positionlist_2017_*.csv positionlist_2020_*.csv \
    --train_min_rate 2500 --test_min_rate 4000 --min_move_num 50 \
    --model model_2017_policy --epoch 10