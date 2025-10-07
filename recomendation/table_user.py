from sqlalchemy import Column, Integer, String
from database import Base



class User(Base):
    __tablename__ = "user"

    age = Column(Integer)
    city = Column(String)
    country = Column(String)
    exp_group = Column(Integer)
    gender = Column(Integer)
    os = Column(String)
    source = Column(String)
    id = Column(Integer, primary_key=True)
