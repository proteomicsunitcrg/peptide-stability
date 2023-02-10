############# Library Import 
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import h5py
from keras.models import model_from_json
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import random  
import re,sys
from keras.layers import Layer, InputSpec
from keras.models import model_from_json
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras import initializers
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.utils import compute_class_weight
from keras.layers import *
from keras.models import *
from keras.regularizers import l1, l2
from keras import initializers, regularizers, constraints
from sklearn.metrics import roc_curve
import matplotlib
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import pickle as cPickle
from keras import initializers, regularizers, constraints
import keras
from keras.layers import Dense, Input,  Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape,  Concatenate, Lambda, Average
from keras.models import Sequential, Model
from keras.initializers import Constant
from keras.layers import add
from sklearn.utils import class_weight
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

################### data_preprocessing
pos_sequences['class'] = '>'  + pos_sequences['class'].astype(str)
neg_sequences['class'] = '>'  + neg_sequences['class'].astype(str)
pos_sequences = pd.concat([pos_sequences['class'] , pos_sequences['Sequence']]).sort_index(kind='merge')
neg_sequences = pd.concat([neg_sequences['class'] , neg_sequences['Sequence']]).sort_index(kind='merge')
data = pd.concat([pos_sequences, neg_sequences])

pool_length = 2
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
 
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
 
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
 
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
 
        if self.bias:
            uit += self.b
 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
 
        a = K.exp(ait)
 
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def trans(str1):
    a = []
    dic = {'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    return a

def createTrainData(str1):
    sequence_num = []
    label_num = []
    f = open(str1).readlines()
    for i in range(0,len(f)-1,2):
        label = f[i].strip('\n').replace('>','')
        label_num.append(int(label))
        sequence = f[i+1].strip('\n')
        sequence_num.append(trans(sequence))

    return sequence_num,label_num

def createTrainTestData(str_path, nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):
    X,labels = cPickle.load(open(str_path, "rb"))

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)
    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                                      'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])


    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    return (X_train, y_train), (X_test, y_test)


max_features = 50
maxlen = 30
embedding_size = 128

# Convolution
#filter_length = 3
nb_filter = 64

lr = 0.0001
# LSTM
lstm_output_size = 70

# Training
batch_size = 64
nb_epoch = 25

a,b = createTrainData(fastafile)
k = []
for i in range(len(a)):
    k.append(a[i] + [len(a[i])+22] + my_data[i])

t = (k, b)
cPickle.dump(t,open(pklfile,"wb"))


(X_train, y_train), (X_test, y_test) = createTrainTestData(pklfile,nb_words=max_features, test_split=0.15)
X_train = pad_sequences(X_train, maxlen=max_len, dtype = 'float32')
X_test = pad_sequences(X_test, maxlen=max_len, dtype = 'float32')

pool_length = 2
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
 
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
 
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def get_config(self):
      config = super().get_config().copy()
      config.update({
          
          'W_regularizer' : self.W_regularizer,
          'u_regularizer' : self.u_regularizer,
          'b_regularizer' : self.b_regularizer,
 
          'W_constraint' : self.W_constraint,
          'u_constraint' : self.u_constraint,
          'b_constraint' : self.b_constraint,
 
          'bias' : self.bias})
      return config    
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
 
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
 
        if self.bias:
            uit += self.b
 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
 
        a = K.exp(ait)
 
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
 
        if self.bias:
            uit += self.b
 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
 
        a = K.exp(ait)
 
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (max_len,))
    
    x = Embedding(max_features, embedding_size, input_length=max_len)(inp)
    x = SpatialDropout1D(dr)(x)
    x = Dropout(0.1)(x)
    x = Bidirectional(GRU(units, return_sequences = True))(x)
    
    #x = Bidirectional(GRU(units, return_sequences = True))(x)
    #x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    x = Bidirectional(GRU(units, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(x)
    x = AttentionWithContext()(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = tf.keras.metrics.BinaryAccuracy(threshold=0.4))
    
#     history = model.fit(X_train, Y_train, batch_size = 128, epochs = 4, validation_data = (X_valid, Y_valid), 
#                         verbose = 1, callbacks = [ra_val, check_point, early_stop])
#     model = load_model(file_path)

    return model

######################################################## k-fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
acc_score = []
auc_score = []
sn_score = []
sp_score = []
mcc_score = []


class_weights = compute_class_weight(class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train)
# class_weights = dict(zip(np.unique(train_Y), class_weights)),
class_weights = {l:c for l,c in zip(np.unique(y_train), class_weights)}

######################################################## k-fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
acc_score = []
auc_score = []
sn_score = []
sp_score = []
mcc_score = []
class_weights = compute_class_weight(class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train)
# class_weights = dict(zip(np.unique(train_Y), class_weights)),
class_weights = {l:c for l,c in zip(np.unique(y_train), class_weights)}
training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience = 2, factor = 0.25, min_lr = 1e-07, verbose = 1),
    keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True)
]
# from imblearn.combine import SMOTETomek
# smt = SMOTETomek(random_state=42)
# X_train, y_train = smt.fit_resample(X_train, y_train)
import collections 
for i,(train, test) in enumerate(kfold.split(X_train, y_train)):
    
    print('\n\n%d'%i)
    model = build_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)
    history = model.fit(X_train[train], y_train[train], class_weight=class_weights,epochs = nb_epoch, batch_size = batch_size 
                        ,callbacks=training_callbacks, validation_data = (X_train[test], y_train[test]), shuffle = True, verbose=2)

    model.save(model_save,overwrite=True)
    ###########################
    pre_test_y = model.predict(X_train[test], batch_size = batch_size)
    print(pre_test_y)
    print('\n')
    vfpr_keras, vtpr_keras, thresholds_keras = roc_curve(y_train[test], pre_test_y)
    val_auc = metrics.roc_auc_score(y_train[test], pre_test_y)
    auc_score.append(val_auc)
    print("val_auc: ", val_auc) 

    score, acc = model.evaluate(X_train[test], y_train[test], batch_size=batch_size)
    acc_score.append(acc)
    print('val score:', score)
    print('val accuracy:', acc)
    print('***********************************************************************\n')
print('***********************print final result*****************************')
print(acc_score,auc_score)
mean_acc = np.mean(acc_score)
mean_auc = np.mean(auc_score)
#print('mean acc:%f\tmean auc:%f'%(mean_acc,mean_auc))

line = 'acc\tauc:\n%.2f\t%.4f'%(100*mean_acc,mean_auc)
print('5-fold result\n'+line)

import joblib
Rmf_model = joblib.load("./random_forest.joblib")

model_XTest = []
mX_test=[]
for i in range(len(X_test)):
    model_XTest.append(X_test[i][-44:])
predict_6points = Rmf_model.predict(model_XTest)

mX_test = []
for i in range(len(model_XTest)):
    mX_test.append(list(model_XTest[i]) + list(predict_6points[i]))
    
    
###########################
auc_score = []
acc_score = []
pre_test_y = model.predict(np.array(mX_test), batch_size = batch_size)
print(pre_test_y)
print('\n')
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, pre_test_y)
test_auc = metrics.roc_auc_score(y_test, pre_test_y)
auc_score.append(test_auc)


score, acc = model.evaluate(np.array(mX_test), y_test, batch_size=batch_size)
acc_score.append(acc)
print('Test score:', score)
print('Test accuracy:', acc)
print("test_auc: ", test_auc) 

from sklearn.metrics import matthews_corrcoef, classification_report
print(classification_report(y_test, pre_test_y.round()))

matthews_corrcoef(y_test, pre_test_y.round())

