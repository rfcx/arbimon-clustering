FROM python:3.7 as arbimon-jobs-clustering

ADD cluster.py /
ADD db.py /

RUN apt-get -y update

RUN pip install numpy scikit-learn boto3 npy-append-array umap-learn pacmap SQLAlchemy==1.4.39 mysql-connector-python-rf
