FROM python:3.8 as arbimon-jobs-clustering

ADD cluster.py /
ADD db.py /

RUN apt-get -y update

RUN pip install numpy scikit-learn boto3 npy-append-array umap-learn pacmap SQLAlchemy==1.4.47 mysql-connector-python==8.0.20
