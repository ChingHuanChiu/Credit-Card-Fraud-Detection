from typing import Dict

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float
from sqlalchemy.orm import sessionmaker



Base = declarative_base()


engine = create_engine("mysql+pymysql://root:CHIOU840509@127.0.0.1:3306/Credict")
session = sessionmaker(bind=engine)()


class Credict(Base):

    __tablename__ = 'clients'

    id = Column(Integer, primary_key=True, autoincrement=True)
    bacno = Column(String(50))
    txkey = Column(String(10000))
    locdt = Column(String(50))
    loctm = Column(String(50))
    cano = Column(String(50))
    contp = Column(String(50))
    etymd = Column(String(50))
    mchno = Column(String(50))
    acqic = Column(String(50))
    mcc = Column(String(50))
    conam = Column(String(50))
    ecfg = Column(String(5))
    insfg = Column(String(5))
    iterm = Column(String(50))
    stocn = Column(String(50))
    scity = Column(String(50))
    stscd = Column(String(50))
    ovrlt = Column(String(5))
    flbmk = Column(String(30))
    hcefg = Column(String(50))
    csmcu = Column(String(50))
    flg_3dsmk = Column(String(30))
    predict = Column(String(5))


def insert_todb(data: Dict[str, str], error_logger) -> None:

    try:
        new_info = Credict(**data)
        session.add(new_info)
        session.commit()
    except Exception as e:
        session.rollback()
        error_logger.info('Error encounter when insert data to db:', e)
    # session.close()
