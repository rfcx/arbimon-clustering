import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# mysql2pg Phase 5.5 (W1, 2026-07-15): dialect-aware connection.
# ARBIMON_DB_DIALECT=mysql (DEFAULT - byte-for-byte today's behavior) or
# postgres. The PG path activates ONLY via env at the coordinated jobs-plane
# flip; nothing changes for existing deploys. Env names stay MYSQL_* for
# both dialects (worker-port template rule 7 - the flip changes env VALUES,
# not names; ARBIMON_DB_PORT overrides MYSQL_PORT when set).
#
# Credentials are URL-quoted (quote_plus): the old string-concat URL broke
# on any password containing URL-special characters (found by the W1 smoke
# against live creds - the SQLAlchemy URL parser mis-split the DSN).

def connect():
    dialect  = os.environ.get('ARBIMON_DB_DIALECT', 'mysql').lower()
    user     = quote_plus(os.environ.get('MYSQL_USERNAME') or '')
    password = quote_plus(os.environ.get('MYSQL_PASSWORD') or '')
    host     = os.environ.get('MYSQL_HOSTNAME')
    schema   = os.environ.get('MYSQL_NAME')
    port     = os.environ.get('ARBIMON_DB_PORT') or os.environ.get('MYSQL_PORT')

    if dialect in ('postgres', 'postgresql', 'pg'):
        url = ('postgresql+psycopg2://' + user + ':' + password + '@'
               + host + ':' + port + '/' + schema)
        engine = create_engine(url, poolclass=NullPool,
                               connect_args={'connect_timeout': 10})
    else:
        url = ('mysql+mysqlconnector://' + user + ':' + password + '@'
               + host + ':' + port + '/' + schema)
        engine = create_engine(url, poolclass=NullPool)
    Session  = sessionmaker(bind=engine, autocommit=False)
    metadata = MetaData()

    return Session(), engine, metadata