import os
from catboost import CatBoostClassifier
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import create_engine, exc
from fastapi import FastAPI
from datetime import datetime
from typing import List
from schema import PostGet, Response
import numpy as np
from tenacity import  retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import hashlib
import struct


def get_model_path(model_name: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = f"/workdir/user_input/{model_name}"
    else:
        if model_name == "model_control":
            MODEL_PATH = "C:/Users/MagPC/Documents/pythonml/startml/catboost_model_control"
        elif model_name == "model_test": 
            MODEL_PATH = "C:/Users/MagPC/Documents/pythonml/startml/catboost_model_test"
    return MODEL_PATH


def load_models():

    model_control_path = get_model_path("model_control")
    model_test_path = get_model_path("model_test")

    model_control = CatBoostClassifier()
    model_control.load_model(model_control_path)

    model_test = CatBoostClassifier()
    model_test.load_model(model_test_path)

    return model_control, model_test


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(exc.OperationalError),
)
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 100000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = None
    try:
        conn = engine.connect().execution_options(stream_results=True)
        chunks = []
        for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
            chunks.append(chunk_dataframe)
        return pd.concat(chunks, ignore_index=True)
    finally:
        if conn:
            conn.close()


def load_features() -> pd.DataFrame:
    user_data_control = batch_load_sql("SELECT * FROM mihail_klentsar_gqf7893_users_22_old")
    user_data_test = batch_load_sql("SELECT * FROM mihail_klentsar_gqf7893_users_22_test")
    return user_data_control, user_data_test


def load_posts():
    posts_control = batch_load_sql("SELECT * FROM mihail_klentsar_gqf7893_posts_22_old")
    posts_test = batch_load_sql("SELECT * FROM mihail_klentsar_gqf7893_posts_22_test")
    post_text_df = batch_load_sql('SELECT * FROM public.post_text_df;')

    return posts_control, posts_test , post_text_df

def load_feed(id):
    return batch_load_sql(f"SELECT * FROM public.feed_action WHERE user_id = {id} and action = 'view'")

SALT = "my_experiment_salt_2025_v2"  
CONTROL_GROUP_PERCENTAGE = 50


def get_exp_group(user_id: int) -> str:
 
    user_str = f"{user_id}{SALT}"

    # Создаем хэш
    hash_object = hashlib.md5(user_str.encode())
    hash_bytes = hash_object.digest()

    hash_int = struct.unpack("<I", hash_bytes[:4])[0]

    # Нормализуем до 0-100
    normalized_value = hash_int % 100

    if normalized_value < CONTROL_GROUP_PERCENTAGE:
        return "control"
    else:
        return "test"


app = FastAPI()


def get_recommendations_control(
    user_id: int, limit: int, model_control, user_data_control, posts_control, post_text_df
) -> List[PostGet]:

    user = user_data_control[user_data_control["user_id"] == user_id].drop(
        "user_id", axis=1
    )
    viewed_posts = []  
    unviewed = posts_control[~posts_control["post_id"].isin(viewed_posts)]

    user_features = (
        user.drop("like_posts", axis=1) if "like_posts" in user.columns else user
    )
    userdf = pd.merge(user_features, unviewed, how="cross")
    post_ids = userdf["post_id"]
    userdf = userdf.drop("post_id", axis=1)

    userdf["pred"] = model_control.predict_proba(userdf)[:, 1]
    userdf["post_id"] = post_ids
    preds = list(userdf.sort_values("pred", ascending=False).head(limit)["post_id"])

    recommendations = []
    for post_id in preds:
        post_info = post_text_df[post_text_df["post_id"] == post_id]
        if not post_info.empty:
            recommendations.append(
                PostGet(
                    id=post_id,
                    text=post_info["text"].values[0],
                    topic=post_info["topic"].values[0],
                )
            )

    return recommendations


def get_recommendations_test(
    user_id: int, limit: int, model_test, user_data_test, posts_test, post_text_df
) -> List[PostGet]:

    user = user_data_test[user_data_test["user_id"] == user_id].drop("user_id", axis=1)

    viewed_posts = []  
    unviewed = posts_test[~posts_test["post_id"].isin(viewed_posts)]

    user_features = (
        user.drop("like_posts", axis=1) if "like_posts" in user.columns else user
    )
    userdf = pd.merge(user_features, unviewed, how="cross")
    post_ids = userdf["post_id"]
    userdf = userdf.drop("post_id", axis=1)

    if "cluster" in userdf.columns and "second_cluster" in userdf.columns:
        userdf[["cluster", "second_cluster"]] = userdf[
            ["cluster", "second_cluster"]
        ].astype("category")

    userdf["pred"] = model_test.predict_proba(userdf)[:, 1]
    userdf["post_id"] = post_ids

    preds = list(userdf.sort_values("pred", ascending=False).head(limit)["post_id"])

    recommendations = []
    for post_id in preds:
        post_info = post_text_df[post_text_df["post_id"] == post_id]
        if not post_info.empty:
            recommendations.append(
                PostGet(
                    id=post_id,
                    text=post_info["text"].values[0],
                    topic=post_info["topic"].values[0],
                )
            )

    return recommendations


user_data_control, user_data_test = load_features()
posts_control,posts_test, post_text_df = load_posts()
model_control, model_test = load_models()


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    exp_group = get_exp_group(id)

    print(f"User {id} is in group: {exp_group}")

    if exp_group == "control":
        recommendations = get_recommendations_control(
            id, limit, model_control, user_data_control, posts_control, post_text_df
        )
    else:  # test group
        recommendations = get_recommendations_test(
            id, limit, model_test, user_data_test, posts_test, post_text_df
        )

    return Response(exp_group=exp_group, recommendations=recommendations)
