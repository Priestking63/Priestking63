import datetime

from pydantic import BaseModel
from typing import List


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class UserGet(BaseModel):
    age : int
    city : str
    country: str
    exp_group: int
    gender: int
    os: str
    source: str
    id : int

    class Config:
        orm_mode = True


class FeedGet(BaseModel):
    user_id : int
    user : UserGet
    post : PostGet
    post_id: int
    action:str
    time: datetime.datetime

    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]
