FROM python:3.6.9

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python-opengl

RUN pip3 install --upgrade pip
COPY pip.packages /tmp/
RUN pip3 install --trusted-host pypi.python.org -r /tmp/pip.packages

# PUT ALL CHANGES UNDER THIS LINE

RUN pip install --upgrade pip

# install chainer from source
RUN git clone https://github.com/chainer/chainer.git
RUN cd chainer && pip install . && cd ..

# install chainerRL from source
RUN git clone https://github.com/chainer/chainerrl.git
RUn cd chainerrl && python setup.py install && cd ..


RUN mkdir /src
WORKDIR /src
ADD . ./

