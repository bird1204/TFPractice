# 20190625 libgirl pre-test
# - 安裝Linux (建議使用虛擬機器或docker, 但因docker設定較為複雜, 可依您對其熟悉程度自由選用)
# - 於Linux 環境中安裝Conda
# - 於conda創建新環境並安裝lgl, 安裝方式請詳: lgl on GitHub (lgl為本團隊為python開發的Launcher開源專案)
# 執行：
# docker build -t python:3.6
# docker run -ti --rm python:3.6

FROM debian:buster

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \ python3.6 python-pip \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

RUN echo "conda version is `conda --version`"
RUN echo "pip version is `pip --version`"

RUN pip install lgl

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN mkdir /workspace
WORKDIR /workspace