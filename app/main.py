from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

import os

os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'

app = FastAPI()


class Document(BaseModel):
    content: str


# https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder-xling-many/1.tar.gz
module_url = "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1"

# Set up graph.
g = tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    xling_8_embed = hub.Module(module_url)
    embedded_text = xling_8_embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Initialize session.
session = tf.Session(graph=g)
session.run(init_op)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post('/infer_vec')
def get_xling_vector(doc: Document):
    result = session.run(embedded_text, feed_dict={text_input: [doc.content]})
    return {'vec': result.tolist()}
