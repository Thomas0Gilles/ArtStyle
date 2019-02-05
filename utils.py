# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:11:46 2018

@author: tgill
"""
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
from keras.utils import to_categorical
from keras import backend as K
import cv2

data = "wikiart/wikiart/"
data_train = r"C:\Users\tgill\OneDrive\Documents\GD_AI\ArtGAN\wikipaintings_full\wikipaintings_train"
data_test = r"C:\Users\tgill\OneDrive\Documents\GD_AI\ArtGAN\wikipaintings_full\wikipaintings_train"
target_size = (224, 224)

def load_samples(class_size, data=data_train):
    X = []
    y = []
    classes = []
    for i, style in enumerate(os.listdir(data)):
        classes.append((i, style))
        path = os.path.join(data, style)
        for pic in os.listdir(path)[:class_size]:
            path_pic = os.path.join(path, pic)
            #image = cv2.imread(path_pic)
            #image = Image.open(path_pic)
            image = img_to_array(load_img(path_pic, target_size = target_size))
            
            X.append(np.ascontiguousarray(image))
            y.append(i)
    return np.ascontiguousarray(X), np.ascontiguousarray(to_categorical(y)), classes

def load_data(data, target_size=None, prep_func=None):
    X = []
    y = []
    classes = []
    for i, style in enumerate(os.listdir(data)):
        classes.append((i, style))
        path = os.path.join(data, style)
        for pic in os.listdir(path):
            path_pic = os.path.join(path, pic)
            #image = cv2.imread(path_pic)
            #image = Image.open(path_pic)
            #img = img_to_array(load_img(path_pic, target_size = target_size))
            img = cv2.imread(path_pic)
            if img is not None:
                img = img[..., ::-1]
                if target_size is not None:
                    img = cv2.resize(img, dsize=target_size)
                if prep_func is not None:
                    img = prep_func(img)
                X.append(np.ascontiguousarray(img))
                y.append(i)
    #X = imagenet_preprocess_input_batch(np.ascontiguousarray(X))
    return np.ascontiguousarray(X), np.ascontiguousarray(to_categorical(y)), classes

def load_data_part(data, target_size=None, p=1, prep_func=None):
    X = []
    y = []
    classes = []
    for i, style in enumerate(os.listdir(data)):
        classes.append((i, style))
        path = os.path.join(data, style)
        for pic in os.listdir(path):
            r = np.random.random_sample()
            if r<p:
                path_pic = os.path.join(path, pic)
                #image = cv2.imread(path_pic)
                #image = Image.open(path_pic)
                #img = img_to_array(load_img(path_pic, target_size = target_size))
                img = cv2.imread(path_pic)
                if img is not None:
                    img = img[..., ::-1]
                    #print("Before " ,np.mean(img))
                    if target_size is not None:
                        img = cv2.resize(img, dsize=target_size)
                    if prep_func is not None:
                        img = prep_func(img)
                    X.append(np.ascontiguousarray(img))
                    y.append(i)
    #X = imagenet_preprocess_input_batch(np.ascontiguousarray(X))
    return np.ascontiguousarray(X), np.ascontiguousarray(to_categorical(y)), classes

def load_data_frac(data, target_size=None, fractions=10, n_frac=0, prep_func=None):
    X = []
    y = []
    classes = []
    for i, style in enumerate(os.listdir(data)):
        
        classes.append((i, style))
        path = os.path.join(data, style)
        n = len(os.listdir(path))
        
        start = n_frac*(n//fractions)
        end = (n_frac+1)*(n//fractions)
        for j, pic in enumerate(os.listdir(path)):
            if j>=start and j<end:
                path_pic = os.path.join(path, pic)
                img = cv2.imread(path_pic)
                if img is not None:
                    img = img[..., ::-1]
                    if target_size is not None:
                        img = cv2.resize(img, dsize=target_size)
                    if prep_func is not None:
                        img = prep_func(img)
                    X.append(np.ascontiguousarray(img))
                    y.append(i)
    return np.ascontiguousarray(X), np.ascontiguousarray(to_categorical(y)), classes

def getPaths(data=data_train):
    X = []
    y = []
    classes = []
    for i, style in enumerate(os.listdir(data)):
        classes.append((i, style))
        path = os.path.join(data, style)
        for pic in os.listdir(path):
            path_pic = os.path.join(path, pic)
            #image = cv2.imread(path_pic)
            #image = Image.open(path_pic)
            X.append(path_pic)
            y.append(i)
    return np.ascontiguousarray(X), np.ascontiguousarray(to_categorical(y)), classes

def evaluate(model, X_paths, y, p, target_size=None):
    npts = len(X_paths)
    n_eval = int(npts*p)
    idxs = np.random.choice(npts, n_eval, replace=False)
    preds=[]
    c=0
    i=0
    f=0
    for i,idx in enumerate(idxs):
        img = cv2.imread(X_paths[idx])
        if img is not None:
            if target_size is not None:
                img = cv2.resize(img, dsize=target_size)
            pred = model.predict(np.ascontiguousarray([img]))
            preds.append(pred[0])
            c+=1
        else:
            preds.append(y[idx])
            f+=1
    a=np.argmax(np.array(preds), axis=1)
    b=np.argmax(y[idxs], axis=1)
    acc = (np.sum(a==b)-f)/c
    return acc

def producer(queue, stop_event, X_paths, y, batch_size, target_size=None, prep_func=None):
    npts = len(X_paths)
    while not stop_event.is_set():
        X_batch=[]
        y_batch=[]
        c = 0
        while c<batch_size:
#            print(c)
            try:
                idx = np.random.randint(0, npts)
                img = cv2.imread(X_paths[idx])
                #img = img_to_array(load_img(X_paths[idx], target_size = target_size))
                if img is not None:
                    img = img[..., ::-1]
                    if target_size is not None:
                        img = cv2.resize(img, dsize=target_size)
                    if prep_func is not None:
                        img = prep_func(img)
                    #Augmentation
                    flip = np.random.rand()
                    if flip<0.5:
                        img = np.flip(img, axis=1)
                    X_batch.append(img)
                    y_batch.append(y[idx])
                    c+=1
            except:
                pass
        #X_batch = imagenet_preprocess_input_batch(np.ascontiguousarray(X_batch))
        queue.put((np.ascontiguousarray(X_batch), np.ascontiguousarray(y_batch)))
        

def count_files(folder):
    s = 0
    for t in list(os.walk(folder)):
        s += len(t[2])
    return s

def imagenet_preprocess_input(x):
#    x = np.ascontiguousarray(x, dtype=np.int16)
    # 'RGB'->'BGR'
#    x = x[:, :, ::-1]
    # Zero-center by mean pixel

    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(K.floatx(), copy=False)
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
#    x /= 127.5
#    x -= 1.
#    x[:, :, 0] -= 104
#    x[:, :, 1] -= 117
#    x[:, :, 2] -= 124
    return x

def imagenet_preprocess_input_batch(X):
#    x = np.ascontiguousarray(x, dtype=float)
    # 'RGB'->'BGR'
    X = X[ :,:, :, ::-1]
    # Zero-center by mean pixel
#    x[:, :, 0] -= 103.939
#    x[:, :, 1] -= 116.779
#    x[:, :, 2] -= 123.68
    X[: ,:, :, 0] -= 104
    X[:, :, :, 1] -= 117
    X[:, :, :, 2] -= 124
    return X



def empty_resnet():
    K.set_image_data_format('channels_last')
    base_model = ResNet50(weights=None,include_top=False,input_shape=(224,224,3))
    features = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=features)
    return model

def _get_weighted_layers(model):
    res=[]
    for layer in model.layers:
        if len(layer.get_weights()) != 0:
            res.append(layer.name)
    return res

def _set_n_retrain(model,n,reinit=False):
    w_layers = _get_weighted_layers(model)
    if reinit:
        empty_model = empty_resnet()

    if n > len(w_layers):
        n == len(w_layers)
    if n>0:
        if reinit:
            for layer, layer_empty in zip(model.layers, empty_model.layers):
                if layer.name in w_layers[-n:]:
                    layer.trainable = True
                    w = layer_empty.get_weights()
                    layer.set_weights(w)
                else:
                    layer.trainable = False
        else :
            for layer in model.layers:
                if layer.name in w_layers[-n:]:
                    layer.trainable = True
                else:
                    layer.trainable = False

    else :
        for layer in model.layers:
            layer.trainable = False

    return model

def categorical_acc(true, pred):
    return np.mean(true==pred)

def mean_acc(true, pred):
    classes = np.unique(true)
    class_acc=[]
    class_prec=[]
    for c in classes:
        pos = (true == c)
        neg = (true != c)
        pos_true = (pred[pos]==c)
        neg_true = (pred[neg]!=c)
        acc = (np.sum(pos_true)+np.sum(neg_true))/len(true)
        prec = np.sum(pos_true)/np.sum(pos)
        class_acc.append(acc)
        class_prec.append(prec)
    return class_acc, class_prec