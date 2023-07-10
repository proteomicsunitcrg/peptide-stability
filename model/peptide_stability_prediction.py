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
              maxlen=None, test_split=0, seed=113,
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
max_features = 30
max_len = 50
embedding_size = 128
fastafile = './My_final_data.csv'
modelfile = "test.txt"
pklfile = "model.pkl"
model_save = 'model_save.h5'
# Convolution
#filter_length = 3
nb_filter = 3

# Training
batch_size = 32
nb_epoch = 50

a,b = createTrainData(fastafile)

t = (a, b)
cPickle.dump(t,open(pklfile,"wb"))

(X_train, y_train), (X_test, y_test) = createTrainTestData(pklfile,nb_words=max_features)
X_train = tf.keras.utils.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.utils.pad_sequences(X_test, maxlen=max_len)

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
    x = Dropout(0.5)(x)
    x = Bidirectional(GRU(units, return_sequences = True, kernel_regularizer=l2(0.0005)))(x)
    x = Dropout(0.5)(x)
    #x = Bidirectional(GRU(units, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform", kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    x = Conv1D(units, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform", kernel_regularizer=l2(0.0005))(x)
    x = Bidirectional(GRU(units, return_sequences = True))(x)
    #x = Dropout(0.5)(x)
    #x = Bidirectional(GRU(units, return_sequences = True, kernel_regularizer=l2(0.0005)))(x)
    #x = Conv1D(86, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(x)
    x = Dropout(0.5)(x)
    x = AttentionWithContext()(x)
    x = Dropout(0.5)(x)
    # x = Dense(1164, kernel_regularizer=l2(0.0005), activation="elu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(100, kernel_regularizer=l2(0.0005), activation="elu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(50, kernel_regularizer=l2(0.0005), activation="elu")(x)
    # x = Dropout(0.5)(x)
    #x = Dense(10, kernel_regularizer=l2(0.0005), activation="elu")(x)
    #x = Dropout(0.5)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss='BinaryCrossentropy', optimizer = tf.keras.optimizers.legacy.Adam(lr = lr, decay = lr_d), metrics='AUC')
    
#     history = model.fit(X_train, Y_train, batch_size = 128, epochs = 4, validation_data = (X_valid, Y_valid), 
#                         verbose = 1, callbacks = [ra_val, check_point, early_stop])
#     model = load_model(file_path)

    return model 

######################################################## k-fold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
acc_score = []
auc_score = []
sn_score = []
sp_score = []
mcc_score = []
tprs = []
history = []
fold_var = 1
save_dir = '/kaggle/working/'
def get_model_name(k):
    return 'model_'+str(k)+'.h5'
mean_fpr = np.linspace(0, 1, 100)

X = X_train
Y = y_train
import collections 
for i,(train, test) in enumerate(kfold.split(X, Y)):
    print('\n\n%d'%i)
    model = build_model(lr = 1e-3, lr_d = 0, units = 86, dr = 0.2)
    class_weights = compute_class_weight(class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train)
    class_weights = {l:c for l,c in zip(np.unique(y_train), class_weights)}
    training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience = 3,monitor = 'val_loss', factor = 0.1, min_lr = 1e-12, verbose = 1),
        keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True),
        keras.callbacks.ModelCheckpoint(save_dir+get_model_name(i), 
                           monitor='val_auc', verbose=1, 
                           save_best_only=True, mode='max')]

    X_train = X[train]
    y_train = Y[train]  # Based on your code, you might need a ravel call here, but I would look into how you're generating your y
    
    X_val = X[test]
    y_val = Y[test]  # See comment on ravel and  y_train


    history.append(model.fit(X_train, y_train, class_weight=class_weights,epochs = nb_epoch, batch_size = batch_size 
                        ,callbacks=training_callbacks, validation_data = (X_val, y_val), shuffle = True, verbose=2))


    ###########################
    prd_acc = model.predict(X_val, batch_size = batch_size)
    pre_acc2 = []
    for i in prd_acc:
        pre_acc2.append(i[0])

    prd_lable =[]
    for i in pre_acc2:
        if i>0.5:
             prd_lable.append(1)
        else:
             prd_lable.append(0)
    prd_lable = np.array( prd_lable)
    obj = confusion_matrix(y_val, prd_lable)
    tp = obj[0][0]
    fn = obj[0][1]
    fp = obj[1][0]
    tn = obj[1][1]
    sn = tp/(tp+fn)
    sp = tn/(tn+fp)
    mcc= (tp*tn-fp*fn)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
    sn_score.append(sn)
    sp_score.append(sp)
    mcc_score.append(mcc)
    ###########################
    pre_test_y = model.predict(X_val, batch_size = batch_size)
    test_auc = metrics.roc_auc_score(y_val, pre_test_y)
    vfpr_keras, vtpr_keras, thresholds_keras = roc_curve(y_val, pre_test_y)
    tprs.append(np.interp(mean_fpr, vfpr_keras, vtpr_keras))
    tprs[-1][0] = 0.0
    
    auc_score.append(test_auc)
    std_auc = np.std(auc_score)
    print("test_auc: ", test_auc) 
    score, acc = model.evaluate(X_val, y_val, batch_size=batch_size)
    acc_score.append(acc)
    print('Test accuracy:', acc)
    print('Test sn:', sn_score)
    print('Test sp:', sp_score)
    print('Test mcc:', mcc_score)
    fold_var + 1
    print('***********************************************************************\n')
