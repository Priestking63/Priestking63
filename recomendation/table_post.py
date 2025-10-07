from sqlalchemy import Column, Integer, Text, String
from database import Base


class Post(Base):
    __tablename__ = "post"

    id = Column(Integer, primary_key=True, name="id")
    text = Column(Text, name="text")
    topic = Column(String, name="topic")
