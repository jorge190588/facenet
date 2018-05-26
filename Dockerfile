FROM nvidia/cuda:9.0-devel-ubuntu16.04

MAINTAINER Jorge Santos

RUN apt-get update 
RUN apt-get --yes install python-pip
RUN pip install tensorflow==1.7
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install opencv-python
RUN pip install h5py
RUN apt-get --yes build-dep python-matplotlib
RUN pip install Pillow
RUN pip install requests
RUN pip install psutil
RUN python -m pip install jupyter

EXPOSE 6006 
EXPOSE 8886 
EXPOSE 8888

CMD jupyter notebook --ip=0.0.0.0 --port=8880 --allow-root