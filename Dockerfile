FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install sounddevice matplotlib numpy librosa opencv-python
