FROM nvidia/cuda:11.1-base-ubuntu18.04 

RUN apt update
RUN	apt install -y apt-utils wget
RUN	apt install -y git python3 python3-pip
RUN apt install -y libsm6 libxrender1
RUN pip3 install -U pip setuptools wheel
RUN pip3 install argparse falcon gunicorn nltk omegaconf opencv-python==4.2.0.34 pydensecrf scikit-image sklearn spacy torch wget
RUN python3 -m spacy download en
RUN git clone https://github.com/WurmD/ImageTextSimilarityApp.git
RUN wget 'https://www.dropbox.com/s/icpi6hqkendxk0m/deeplabv2_resnet101_msc-cocostuff164k-100000.pth?raw=1' -O ImageTextSimilarityApp/deeplabpytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth

WORKDIR /ImageTextSimilarityApp

# --gpus=all requires docker run (docker start does not support it), ergo, can't use CMD
# CMD gunicorn image_similarity_app
