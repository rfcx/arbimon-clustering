import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool


def connect():

    # MySQL: establish connection
    user     = os.environ.get('MYSQL_USERNAME')
    password = os.environ.get('MYSQL_PASSWORD')
    host     = os.environ.get('MYSQL_HOSTNAME')
    schema   = os.environ.get('MYSQL_NAME')
    port     = os.environ.get('MYSQL_PORT')
    engine   = create_engine('mysql+mysqlconnector://' + user + ':' + password + '@' + host + ':' + port + '/' + schema, poolclass=NullPool)
    Session  = sessionmaker(bind=engine, autocommit=False)
    metadata = MetaData()

    return Session(), engine, metadata
