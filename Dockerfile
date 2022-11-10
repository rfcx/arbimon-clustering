FROM python:3.7

ADD cluster.py /
ADD db.py /

RUN apt-get -y update

RUN pip install numpy sklearn boto3 npy-append-array umap-learn pacmap SQLAlchemy mysql-connector-python-rf
