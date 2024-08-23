import configparser
from tensorflow.keras import models
import pickle
import tensorflow as tf
import numpy as np


def load_model():
    model = models.load_model(f"../vol/model.h5")
    cv = pickle.load(open(f"../vol/cv.pkl", "rb"))
    return model, cv

def get_config_variables():
    config = configparser.ConfigParser()
    config.read("../vol/config.ini")
    
    kmer_size = config.getint("encoding", "kmer_size")
    kmer_step = config.getint("encoding", "kmer_step") 
    th_str = config.get("encoding", "th")
    th = int(th_str) if th_str.isdigit() else None
    
    return kmer_size, kmer_step, th

def get_kmers(seq, size=3, step=1):
    return [seq[x:x+size].lower() for x in range(0, len(seq) - size + 1, step)]
    
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    X = tf.SparseTensor(indices, coo.data, coo.shape)
    return tf.sparse.reorder(X)
