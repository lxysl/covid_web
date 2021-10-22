FROM python:3.7-slim-stretch

ADD requirements.txt /
RUN pip --default-timeout=500 install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip --default-timeout=500 install git+https://github.com/frgfm/torch-cam.git#egg=torchcam

ADD . /app
WORKDIR /app

EXPOSE 5000
CMD [ "python" , "app.py"]
