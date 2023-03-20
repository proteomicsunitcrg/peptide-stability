from io import StringIO
from Bio import SeqIO
import pandas as pd
import streamlit as st
import h5py


from PIL import Image
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential, save_model, load_model
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences

import tensorflow as tf
import numpy as np
import protutil
import base64
import joblib 


st.set_page_config(page_title='Peptide Stability prediction')
st.title("Peptide Stability prediction")


alphabet = "XACDEFGHIKLMNOPQRSTUVWY"



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

# import tensorflow as tf
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
def seqValidatorProtein (seq):
    alphabet = "ACDEFGHIKLMNOPQRSTUVWY"
    for i in range(len(seq)):
        if (seq[i] not in alphabet):
            return False
    return True    

# load Model For Gender Prediction

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



model = load_model(
    'model_save.h5', 
    custom_objects={'AttentionWithContext' : AttentionWithContext})




st.sidebar.subheader(("Abstract"))
st.sidebar.markdown("In proteomics peptides are used as surrogates for protein quantification and therefore peptides selection is crucial for good protein quantification.At the same time large-scare proteomics studies involve hundreds of samples and take several weeks of measurement time. Thefore, the study of peptide stability during the duration of the project is essencial for quantification results with high accuracy and precision. The goal of this webserver is to predict the stability of peptides.")





st.sidebar.subheader(("Please Read requirements"))


st.sidebar.markdown("- Please Enter Valid amino acid letters (ACDEFGHIKLMNOPQRSTUVWY)")
st.sidebar.markdown("- Please Enter only peptide of length <20")
st.sidebar.markdown("- No Post-translational modifications are supported")

#st.sidebar.text_area("Sequence Input", height=200)


seq = ""
len_seq = 0

caption= "The proposed methodology to develop Peptide/Protein Stability classifier"
#st.image(image, use_column_width=True, caption=caption)

st.subheader(("Input Sequence(s)"))
#fasta_string  = st.sidebar.text_area("Sequence Input", height=200)
seq_string = st.text_area("Ex: LAENVKIK", height=200)          



if st.button("PREDICT"):
    if (len(seq_string)>30):
        st.error("Please input the sequence between 7 and 30 character")
        exit()
    if (seq_string==""):
        st.error("Please input the sequence first")
        exit()
    
    #validation of proper protein string
    if (not seqValidatorProtein(seq_string)):
        st.error("Invalid Protein Sequence detected. Remove numerics and special characters")
        exit()
    
    else:
        

        def trans(str1):
            a = []
            dic = {'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}
            for i in range(len(str1)):
                a.append(dic.get(str1[i]))
            return a

        Sequence = [trans(seq_string)]
        Sequence = tf.keras.preprocessing.sequence.pad_sequences(Sequence, maxlen=50)
        Sequence_y = model.predict(Sequence)

        def predict_prob(number):
            return [number[0],1-number[0]]

        predict_probability = np.array(list(map(predict_prob, Sequence_y)))
        
        
        

        
        
        if predict_probability[0][0] > 0.5:
            st.subheader('Peptide {} is unstable with a probability of {}%'.format(seq_string , round(predict_probability[0][0]*100)))
        else:
            st.subheader('Peptide {} is stable with a probability of {}%'.format(seq_string , round(predict_probability[0][1]*100)))


   



    #final_df = pd.DataFrame({'Stability prediction': tmp })          
    #st.table(final_df) #does not truncate columns    
    
