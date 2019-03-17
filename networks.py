# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:20:09 2018

@author: tgill
"""
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Lambda
from keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Flatten, Concatenate, Permute, BatchNormalization, Dropout, ReLU, Add
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.utils import to_categorical
from keras_layer_normalization import LayerNormalization
from utils import load_samples, count_files, evaluate, imagenet_preprocess_input, _set_n_retrain, _get_weighted_layers, producer, getPaths, load_data, load_data_part
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
import random

def gram_matrix(x):
    x_ = K.permute_dimensions(x, (0, 3, 1, 2))
    features = tf.reshape(x_, [tf.shape(x_)[0], tf.shape(x_)[1], tf.reduce_prod(tf.shape(x_)[2:])])
    #gram = K.batch_dot(features, K.transpose(features))
    gram = K.batch_dot(features, features, axes=[2,2])
    return gram


def resnet_trained(n_retrain_layers = 0):
    K.set_image_data_format('channels_last')
    K.set_learning_phase(0)
    base_model = ResNet50(include_top=False,input_shape=(224,224,3))
    features = GlobalAveragePooling2D()(base_model.output)
    res = Model(inputs=base_model.input, outputs=features)
    res = _set_n_retrain(res,n_retrain_layers)
    K.set_learning_phase(1)
    out = Dense(units=25, activation='softmax')(res.output)
    model = Model(inputs=res.input, outputs=out)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_layers(model, idxs):
    out=[]
    for i in idxs.tolist():
        out.append(model.layers[i].output)
    m = Model(model.input, out)
    return m

def select(x, n):
    #idxs = [i for i in range(n*n) if i%n<=i//n]
    ones = tf.ones((n,n))
    mask_a = tf.matrix_band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool) # Make a bool mask
    y = tf.boolean_mask(x, mask, axis=1)
    y = tf.math.l2_normalize(y, axis=-1)
    return y

def gram_layer(n):
    vgg = VGG19(include_top=False, weights='imagenet')
    out = vgg.layers[n].output
    m = Model(inputs=vgg.input, outputs=out)
    return m

def resnet_layer(n_layer=-1, n_retrain_layers = 0):
    K.set_image_data_format('channels_last')
    K.set_learning_phase(0)
    base_model = ResNet50(include_top=False,input_shape=(224,224,3))
    res = Model(inputs=base_model.input, outputs=base_model.layers[n_layer].output)
    res = _set_n_retrain(res,n_retrain_layers)
    K.set_learning_phase(1)
    return res

def resnet_try(n_retrain_layers = 0):
    res =  resnet_layer(n_layer=-1, n_retrain_layers = 0)
    x = GlobalAveragePooling2D()(res.output)
    out = Dense(units=25, activation='softmax')(x)
    
    model = Model(inputs=res.input, outputs=out)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

    
def simple_gram(n=0):

    
    
    #resnet :18 , 28 , 38 , 50 , 60 , 70 , 80 , 92  , 102 , 112 , 122 , 132 , 142 , 154 , 164 , 174 , 
    #tailles 256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048,
    res = resnet_layer(174, n_retrain_layers = n)
    resout = res.layers[92].output
    
    n=64
    #x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(filters=n, kernel_size=(1, 1), padding='same')(resout)
    #x = LayerNormalization()(x)
    x = ReLU()(x)
    #x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(x)
    gram = Lambda(lambda x : gram_matrix(x), output_shape=((n, n)))(x)

    x = Lambda(lambda x : select(x, n), output_shape=(n*(n-1)//2,))(gram)

    x = Dense(128, activation='relu')(x)
    content = res.output
    #content = LayerNormalization()(content)
    content = GlobalAveragePooling2D()(content)
    x = Concatenate()([x, content])
    #x=Dropout(0.2)(x)
    #x = Dense(64, activation='relu')(x)
   
    out = Dense(25, activation='softmax')(x)
    #m = Model(inputs=inp, outputs=out)
    m = Model(inputs=res.input, outputs=out)
    m.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def autoencoder(n, k):
    inp = Input((n*(n-1)//2,))
    x = Dense(k, activation='relu')(inp)
    out = Dense(n*(n-1)//2)(x)
    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer='rmsprop', loss='mean_squared_error')
    return m

def encoder(autoencode):
    code = autoencode.layers[1].output
    return Model(inputs=autoencode.input, outputs=code)

def pipe(gramnet, encode):
    inp = Input((None, None, 3))
    gramnet.trainable=False
    gram = gramnet(inp)
    coder = encoder(encode)
    coder.trainable=False
    code = coder(gram)
    x = Dense(128, activation='relu')(code)
    out = Dense(25, activation='softmax')(x)
    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return m
    

def vgg_gram(k):
    #inp = Input((None, None, 3))
    vgg = VGG19(include_top=False, weights='imagenet')
    idxs = np.array([1, 4, 7, 12, 17])
    m_vgg = get_layers(vgg, idxs[k])
    out = []
    for o in m_vgg.outputs:
        gram = Lambda(lambda x : gram_matrix(x), output_shape=((512, 512)))(o)
        gram = Lambda(lambda x : select(x, 512), output_shape=((130816,)))(gram)
        #gram = Flatten()(gram)
        out.append(gram)
    if len(out)>1:
        out = Concatenate()(out)
    model = Model(inputs=vgg.input, outputs=out)
    return model
    
    
class GramFeat:
    def __init__(self, gram_net, ):
        self.gram = gram_net
        
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return self.gram.predict(X, batch_size=64)
    
    def fit_transform(self, X, y):
        return self.gram.predict(X, batch_size=64)
    
def gram_svm():
    return Pipeline([('Gram', GramFeat()), ('SVM', LinearSVC(verbose=1))])

def gram_sgd(gram_net, max_iter=10):
    return Pipeline([('Gram', GramFeat(gram_net)), ('SVM', SGDClassifier(verbose=0, tol=1E-2, n_jobs=4, max_iter=max_iter, loss='log'))])

def gram_clf(gram_net, clf):
    return Pipeline([('Gram', GramFeat(gram_net)), ('CLF', clf)])


def gram_svm_vote(svms):
    return Pipeline([('Gram', GramFeat()), ('Vote', VotingClassifier(svms))])

class GramPipeline:
    def __init__(self, clf = SGDClassifier()):
        self.gram = vgg_gram([3])
        self.clf = clf
    
    def partial_fit(self, X, y, classes, n_iter=20):
        X = self.gram.predict(X, batch_size=64)
        for i in range(n_iter):
            #shuffle_idx = random.shuffle(list(range(len(X))))
            shuffle_idx = np.random.permutation(len(X))
            self.clf.partial_fit(X[shuffle_idx], y[shuffle_idx], classes)
        #self.clf.partial_fit(X, y, classes)
        return self
    
    def fit(self, X, y):
        X = self.gram.predict(X, batch_size=64)
        self.clf.fit(X, y)
        return self
    
    def predict(self, X):
        X = self.gram.predict(X, batch_size=64)
        pred = self.clf.predict(X)
        return pred
    
class GramVote:
    def __init__(self, gram_net, clfs):
        self.gram = gram_net
        self.clfs = clfs
        
    def predict(self, X):
        X = self.gram.predict(X, batch_size=64)
        preds=[]
        preds_proba=[]
        for clf in self.clfs:
            pred = clf.predict(X)
            pred_c = to_categorical(pred, num_classes=25)
            preds.append(pred_c)
            pred_proba = clf.predict_proba(X)
            preds_proba.append(pred_proba)
        pred_sum = np.sum(preds, axis=0)
        pred_proba_sum = np.sum(preds_proba, axis=0)
        return np.argmax(pred_sum, axis=1), np.argmax(pred_proba_sum, axis=1)
    
    def proba_predict(self, X):
        X = self.gram.predict(X, batch_size=64)
        preds=[]
        for clf in self.clfs:
            pred = clf.predict_proba(X)
            preds.append(pred)
        pred_sum = np.sum(preds, axis=0)
        return np.argmax(pred_sum, axis=1)
    
class Vote:
    def __init__(self, clfs):
        self.clfs = clfs
        
    def predict(self, X):
        preds=[]
        preds_proba=[]
        for clf in self.clfs:
            pred = clf.predict(X)
            pred_c = to_categorical(pred, num_classes=25)
            preds.append(pred_c)
            pred_proba = clf.predict_proba(X)
            preds_proba.append(pred_proba)
        pred_sum = np.sum(preds, axis=0)
        pred_proba_sum = np.sum(preds_proba, axis=0)
        return np.argmax(pred_sum, axis=1), np.argmax(pred_proba_sum, axis=1)
    
    def proba_predict(self, X):
        preds=[]
        for clf in self.clfs:
            pred = clf.predict_proba(X)
            preds.append(pred)
        pred_sum = np.sum(preds, axis=0)
        return np.argmax(pred_sum, axis=1)
    
    def predict_proba(self, X):
        preds=[]
        for clf in self.clfs:
            pred = clf.predict_proba(X)
            preds.append(pred)
        pred_sum = np.sum(preds, axis=0)
        return pred_sum
    
class VoteHard:
    def __init__(self, clfs):
        self.clfs = clfs
        
    def predict(self, X):
        preds=[]
        for clf in self.clfs:
            pred = clf.predict(X)
            pred_c = to_categorical(pred, num_classes=25)
            preds.append(pred_c)
        pred_sum = np.sum(preds, axis=0)
        return np.argmax(pred_sum, axis=1), np.argmax(pred_sum, axis=1)
    
        
        
def gram_pca():
    return Pipeline([('Gram', GramFeat()), ('PCA', PCA(n_components=4096, svd_solver='randomized')), ('SVM', LinearSVC(verbose=1))])

def reform(x):
    x_ = K.permute_dimensions(x, (0, 3, 1, 2))
    features = tf.reshape(x_, [tf.shape(x_)[0], tf.shape(x_)[1], tf.reduce_prod(tf.shape(x_)[2:]), 1])
    img = tf.image.resize_images(features, (224, 224))
    return img

def learned_gram():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg_m = get_layers(vgg, np.array([17]))
    for layer in vgg_m.layers:
        layer.trainable=False
    inp = Input((None, None, 3))
    x = vgg_m(inp)
    x = Lambda(lambda x : reform(x))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(19, 19), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(25, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
def tan_vgg():
    vgg = VGG19(include_top=True, weights='imagenet')
    for layer in vgg.layers:
        layer.trainable=False
    inp = Input((224, 224, 3))
    x = vgg(inp)
    out = Dense(25, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
