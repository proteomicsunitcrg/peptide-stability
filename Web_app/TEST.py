import joblib
from io import StringIO
from Bio import SeqIO
import pandas as pd
import numpy as np 
from PIL import Image
import _pickle as cPickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np



model = load_model("Model", compile = True)
sequence = 'LENER'

def trans(str1):
    a = []
    dic = {'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    return a

X = np.array([ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12, 7, 12,7,12])
      
#print(type(X))
#X_test = sequence.pad_sequences(X, maxlen=30)
#print(X_test)

model.predict(X)
