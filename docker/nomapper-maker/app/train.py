#!/usr/bin/python3

"""train.py
Script to train the model.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use CPU only
import argparse
import sys
import pysam
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score


def prepare_class_df(filename, class_num, th):
    seq = []
    try:
        # the whole sequence from left to right
        if th == "none":
            with pysam.AlignmentFile(filename, 'rb') as f:
                for line in f:
                    seq.append(line.seq)        
        # the whole sequence from right to left
        elif th == "reverse_none":
            with pysam.AlignmentFile(filename, 'rb') as f:
                for line in f:
                    rev = "".join(reversed( line.seq ))
                    seq.append(rev)
        # first th nucleotides	
        elif int(th) > 0:   
            th = int(th)
            with pysam.AlignmentFile(filename, 'rb') as f:
                for line in f:
                    seq.append(line.seq[:th])        
        # last th nucleotides
        elif int(th) < 0: 
            th = int(th) 
            with pysam.AlignmentFile(filename, 'rb') as f:
                for line in f:
                    rev = "".join(reversed( line.seq[th:] ))
                    seq.append(rev)
        else:
            print("[ERROR] Parameter 'th' invalid.")
            sys.exit(1)

        df = pd.DataFrame()
        df["seq"] = seq
        df["class"] = class_num
    except ValueError as e:
        print("[ERROR] ", e)
        sys.exit(1)

    return df

def prepare_df(gene_file, no_gene_file, encoding_cfg, test_split_ratio):
    # gene df
    df_gene = prepare_class_df(gene_file, 1, encoding_cfg["th"])
    if encoding_cfg['n_gene'] > df_gene.shape[0]:
        print(f"[WARNING] There are not that many sequences. The n_gene is too large, so it is set to a maximum value = {df_gene.shape[0]}.")
        encoding_cfg['n_gene'] = df_gene.shape[0]
    elif (encoding_cfg['n_gene'] == -1) or (encoding_cfg['n_gene'] == df_gene.shape[0]):
        pass
    else:
        df_gene = df_gene.sample(n=encoding_cfg['n_gene'], replace=False, random_state=42)
    
    test_df_rows = int( test_split_ratio * len(df_gene) )
    test_df_gene = df_gene.sample(n=test_df_rows, replace=False, random_state=42)
    df_gene = df_gene.drop(test_df_gene.index)

    # no-gene df
    df_no_gene = prepare_class_df(no_gene_file, 0, encoding_cfg["th"])
    if (encoding_cfg['n_no_gene'] == -1):
        encoding_cfg['n_no_gene'] = 3 * len(df_gene)
        
    if encoding_cfg['n_no_gene'] > df_no_gene.shape[0]:
        print(f"[WARNING] There are not that many sequences. The n_no_gene is too large, so it is set to a maximum value = {df_no_gene.shape[0]}.")
    elif encoding_cfg['n_no_gene'] == df_gene.shape[0]:
        pass
    else:
        df_no_gene = df_no_gene.sample(n=encoding_cfg['n_no_gene'], replace=False, random_state=42)

    test_df_rows = int( test_split_ratio * len(df_no_gene) )
    test_df_no_gene = df_no_gene.sample(n=test_df_rows, replace=False, random_state=42)
    df_no_gene = df_no_gene.drop(test_df_no_gene.index)

    # concat
    df_all = pd.concat([df_no_gene, df_gene], ignore_index=True)
    test_df_all = pd.concat([test_df_no_gene, test_df_gene], ignore_index=True)
    pickle.dump(test_df_all, open('/vol/test_df.pkl', "wb"))
    summary = df_all['class'].value_counts().sort_index()

    return df_all, summary

def get_kmers(seq, size, step):
    return [seq[x:x+size].lower() for x in range(0, len(seq) - size + 1, step)]

def prepare_input_and_label(df):
    df_texts = list(df['words'])
    for item in range(len(df_texts)):
        df_texts[item] = ' '.join(df_texts[item])
    y = df.iloc[:, 0].values     
    return df_texts, y

def prepare_model(n_words):
    model = models.Sequential()
    model.add( layers.Dense(50, input_shape=(n_words,), activation='relu') )
    model.add( layers.Dense(50, activation='relu') )
    model.add( layers.Dense(1, activation='sigmoid') )
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    X = tf.SparseTensor(indices, coo.data, coo.shape)
    return tf.sparse.reorder(X)

def train_model(model, X, y, e):
    model.fit(X, y, epochs=e, verbose=2)
    model.save('/vol/model.h5')
    return model

def perform_calcs(gene_file, no_gene_file, epochs, encoding_cfg, test_split_ratio):
    df_train, summary = prepare_df(gene_file, no_gene_file, encoding_cfg, test_split_ratio)

    print("Class balance:")
    print(summary)

    # change seq to words
    df_train['words'] = df_train.apply(lambda x: get_kmers(x['seq'], encoding_cfg['kmer_size'], encoding_cfg['kmer_step']), axis=1)
    df_train = df_train.drop('seq', axis=1)

    # prepare X and y
    X, y = prepare_input_and_label(df_train)
    df_train = pd.DataFrame()    

    cv = CountVectorizer(ngram_range=encoding_cfg['ngram_range'])
    X = cv.fit_transform(X)
    pickle.dump(cv, open("/vol/cv.pkl", "wb"))
    
    coded_words = str(list(cv.vocabulary_)[0:5])
    print(f"First 5 encoded words = {coded_words}")

    # train model
    model = prepare_model(X.shape[1])
    X = convert_sparse_matrix_to_sparse_tensor(X)
    model = train_model(model, X, y, epochs)
    
    return model, cv

def test_model(model, cv, encoding_cfg):
    df_test = pickle.load(open(f'/vol/test_df.pkl', "rb")) 
    os.remove('/vol/test_df.pkl')
    
    df_test['words'] = df_test.apply(lambda x: get_kmers(x['seq'], encoding_cfg['kmer_size'], encoding_cfg['kmer_step']), axis=1)
    df_test = df_test.drop('seq', axis=1)
    X, y = prepare_input_and_label(df_test)
    df_test = pd.DataFrame() 
    X = cv.transform(X)
    X = convert_sparse_matrix_to_sparse_tensor(X)
    y_pred = (model.predict(X) > 0.5).astype("int32").reshape((-1,))
        
    conf_mat = pd.crosstab(pd.Series(y, name='Actual'), pd.Series(y_pred, name='Predicted'))
    bal_accuracy = balanced_accuracy_score(y, y_pred)
    return conf_mat, bal_accuracy
    
def parse_tuple(value):
    try:
        parts = value.split(',')
        if len(parts) != 2:
            raise ValueError("The value is not a two-element tuple.")
        return tuple(map(int, parts))
    except ValueError:
        raise argparse.ArgumentTypeError("Incorrect two-element tuple.")

def main():
    parser = argparse.ArgumentParser(description='Script to train the model.')
    parser.add_argument('--gene_file', type=str, default='/vol/mapped.bam', help='the path of the input file (.bam) containing the gene sequences')
    parser.add_argument('--no_gene_file', type=str, default='/vol/unmapped.bam', help='the path of the input file (.bam) containing the no-gene sequences')
    parser.add_argument('--n_gene', type=int, default='-1', help='the number of gene sequences in the dataset (-1 means all available)')
    parser.add_argument('--n_no_gene', type=int, default='-1', help='the number of no-gene sequences in the dataset (-1 means 3x more than all available n_gene)')
    parser.add_argument('--th', default="none", help='the number of nucleotides encoded ("none" - all from left to right, "reverse_none" - all from right to left)')
    parser.add_argument('--kmer_size', type=int, default='3', help='the word size')
    parser.add_argument('--kmer_step', type=int, default='1', help='the word step')
    parser.add_argument('--ngram_range', type=parse_tuple, default=(3, 3), help='the collocation of words subject to encoding (two numbers separated by a comma without a space, e.g. 1,1)')
    parser.add_argument('--epochs', type=int, default=10, help='the number of epochs')
    parser.add_argument('--test_split_ratio', type=float, default=0.1, help='the factor for splitting the dataset into a test set')
    args = parser.parse_args()
    
    encoding_cfg = dict()
    encoding_cfg['n_gene'] = args.n_gene 
    encoding_cfg['n_no_gene'] = args.n_no_gene 
    encoding_cfg['th'] = args.th
    encoding_cfg['ngram_range'] = args.ngram_range
    encoding_cfg['kmer_size'] = args.kmer_size
    encoding_cfg['kmer_step'] = args.kmer_step
    
    model, cv = perform_calcs(args.gene_file, args.no_gene_file, args.epochs, encoding_cfg, args.test_split_ratio)
    conf_mat, bal_accuracy = test_model(model, cv, encoding_cfg)
    NPV = ( conf_mat.iloc[0,0] / (conf_mat.iloc[0,0] + conf_mat.iloc[1,0]) )
    print(f'NPV = {NPV}')
    print(f'bACC = {bal_accuracy}')
    print(f'conf_mat: \n{conf_mat}')

if __name__ == '__main__':
    main()
