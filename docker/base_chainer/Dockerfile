FROM nvidia/cuda:11.0.3-devel-ubuntu18.04

WORKDIR /root

ENV DEBIAN_FRONTEND noninteractive

# Cudaのインストール
COPY libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb /root/

# 必要パッケージのインストール
RUN apt-get update  \
    && apt-get -y upgrade \
    && apt-get install -y \
        git \
        python3 \
        python3-pip \
    && dpkg -i *.deb \
    && apt-get autoremove -y \
    && apt-get clean  \
    && rm -r /var/lib/apt/lists/*

# pythonパッケージのインストール
RUN pip3 install -U pip \
    && pip install --no-cache-dir \
        'cupy-cuda110>=7.7.0,<8.0.0' \
        chainer \
        python-shogi

# 棋譜ファイルの追加
ADD floodgate/wdoor2017.tar.gz /root/floodgate/

# python dlshogiのダウンロード
RUN git clone https://github.com/yanyan0207/python-dlshogi.git
WORKDIR /root/python-dlshogi
RUN git checkout v0.0.1
WORKDIR /root/

# 前処理
RUN python3 python-dlshogi/utils/filter_csa.py floodgate/wdoor2017
RUN python3 python-dlshogi/utils/make_kifu_list.py floodgate/wdoor2017/ kifulist_2017

WORKDIR /root/python-dlshogi
RUN git fetch && git checkout bd7a1d31cd5034ba055c66397acb77dc9e00303e
WORKDIR /root/

# policy
RUN mkdir model
#RUN python3 python-dlshogi/train_policy.py --log model/output_policy.txt kifulist_2017_train.txt kifulist_2017_test.txt
#RUN python3 python-dlshogi/train_value.py -i 1  kifulist_2017_train.txt kifulist_2017_test.txt | tee model/output_value.txt
RUN python3 python-dlshogi/train_policy_value_resnet.py kifulist_2017_train.txt kifulist_2017_test.txt 
RUN mv model/ model_2017/

