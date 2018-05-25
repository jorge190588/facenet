FROM nvidia/cuda:9.0-devel-ubuntu16.04

MAINTAINER Jorge Santos

COPY ./keyboard /etc/default/keyboard
RUN apt-get update 
RUN apt-get install -y git
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get --yes install python-pip 
RUN apt-get install wget
RUN pip install --upgrade pip
RUN pip install --upgrade python
RUN apt-get --yes --force-yes install python-pip
RUN pip install numpy
RUN pip install scipy
RUN python -m pip install --user numpy scipy
RUN pip install plotly
RUN pip install tflearn
RUN pip install asq
RUN apt-get install python-pandas -y
RUN apt-get update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
RUN pip install -U scikit-learn
RUN apt-get install python-opencv -y
RUN apt-get update
RUN pip install tensorflow==1.4
RUN python -m pip install jupyter

EXPOSE 6006 
EXPOSE 8886 
EXPOSE 8888

CMD jupyter notebook --ip=0.0.0.0 --port=8880 --allow-root