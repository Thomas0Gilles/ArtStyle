# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 09:18:06 2018

@author: tgill
"""
import numpy as np

import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg
from keras.preprocessing .image import ImageDataGenerator
from keras import backend as K

from time import time

import queue
import threading
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

from utils import load_samples, count_files, evaluate, imagenet_preprocess_input, _set_n_retrain, _get_weighted_layers, producer, getPaths, load_data, load_data_part, categorical_acc, load_data_frac
from networks import simple_gram

#Must modify the paths so that they point to your local wikipaintings files
data = "wikiart/wikiart/"
data_train = r"C:\Users\tgill\OneDrive\Documents\GD_AI\ArtGAN\wikipaintings_full\wikipaintings_train"
data_test = r"C:\Users\tgill\OneDrive\Documents\GD_AI\ArtGAN\wikipaintings_full\wikipaintings_val"
target_size = (224, 224)



X, y, classes = load_samples(10)
X_test, y_test, classes_test = load_samples(10, data_test)


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

#m = resnet_trained(20)
m = simple_gram(20)

print(m.summary())

n_files_train = count_files(data_train)
n_files_test = count_files(data_test)

nepochs = 20
epoch_size = 2500
batch_size = 32
steps_per_epoch = (n_files_train//batch_size)
v_step = 50 #n_files_test//batch_size
distortions=0.1


train_paths, y_train, classes = getPaths(data_train)
test_paths, y_test, classes = getPaths(data_test)

prep_func = imagenet_preprocess_input

t=time()
X_val, y_val, classes = load_data(data_test, target_size=target_size, prep_func=prep_func)
X_tr, y_tr, classes = load_data_part(data_train, target_size=target_size, p=0.05, prep_func=prep_func)
print("Loaded ", time()-t)

#Threads for asynchronous loading and preprocessing of the images
q = queue.Queue(maxsize=300)
stop_event = threading.Event()
writer = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer2 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer3 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer4 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer5 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer6 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer7 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer8 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer9 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer10 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer11 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer12 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))

writer.start()
writer2.start()
writer3.start()
writer4.start()
writer5.start()
writer6.start()
writer7.start()
writer8.start()
writer9.start()
writer10.start()
writer11.start()
writer12.start()

#Main loop
losses=[]
history = {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[], 'tr_acc':[], 'tr_loss':[]}
print('Start training')
for epoch in range(1, nepochs+1):
    print('Epoch ', epoch)
    loss_epoch=[]
    
    for batch_idx in tqdm(range(1, steps_per_epoch+1)):
    
        X_batch, y_batch = q.get()
        loss=m.train_on_batch(x=X_batch, y=y_batch)
        loss_epoch.append(loss)
    
    loss_epoch = np.asarray(loss_epoch)
    
    tr_loss = np.mean(loss_epoch[:,0])
    tr_acc = np.mean(loss_epoch[:,1])
    print("Train : ", tr_loss, tr_acc)
    
    val_loss, val_acc = m.evaluate(X_val, y_val, batch_size)
  
    loss_tr, acc_tr = m.evaluate(X_tr, y_tr, batch_size)
    print('Other train : ', loss_tr, acc_tr)
    print("Val : ", val_loss,  val_acc)
   
    history['loss'].append(tr_loss)
    history['acc'].append(tr_acc)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    history['tr_loss'].append(loss_tr)
    history['tr_acc'].append(acc_tr)
    
stop_event.set()


