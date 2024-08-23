from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from timeit import default_timer as timer
import app_utils
import numpy as np


MODEL = None
CV = None
KMER_SIZE = None
KMER_STEP = None
TH = None


class SingleSeq(BaseModel):
    seq: str
  
class SeqFile(BaseModel):
    filename: str
    
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load required resources at application start-up
    global MODEL, CV, KMER_SIZE, KMER_STEP, TH
    MODEL, CV = app_utils.load_model()
    KMER_SIZE, KMER_STEP, TH = app_utils.get_config_variables()
    yield
    # Release resources before turning off the application
    MODEL, CV = None, None
    
app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return {'title': 'noMapper API'}

@app.post("/predict/")
def predict(item: SingleSeq):    
    seq = item.seq
    if TH is not None:
        seq = seq[:TH]
    
    words = app_utils.get_kmers(seq, KMER_SIZE, KMER_STEP)
    X = [' '.join(words)]
    X = CV.transform(X)
    X = app_utils.convert_sparse_matrix_to_sparse_tensor(X)
    y_pred = (MODEL.predict(X, verbose=0) > 0.5).astype("int32").reshape((-1,))
    
    answer = 'found' if y_pred[0] == 1 else 'not found'
    return {"result": answer}
