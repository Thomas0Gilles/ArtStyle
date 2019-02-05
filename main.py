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
from networks import simple_gram, gram_pca, gram_svm, learned_gram, tan_vgg, GramPipeline, gram_svm_vote, gram_sgd, GramVote, vgg_gram, gram_clf, Vote, VoteHard, autoencoder, pipe, resnet_try
data = "wikiart/wikiart/"
data_train = r"C:\Users\tgill\OneDrive\Documents\GD_AI\ArtGAN\wikipaintings_full\wikipaintings_train"
data_test = r"C:\Users\tgill\OneDrive\Documents\GD_AI\ArtGAN\wikipaintings_full\wikipaintings_val"
target_size = (224, 224)



X, y, classes = load_samples(10)
X_test, y_test, classes_test = load_samples(10, data_test)

def model(n):
    inp = Input((None, None, 3))
    
#    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
#    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
#    x = MaxPool2D(pool_size=(2, 2))(x)
#    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
#    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
#    x = GlobalAveragePooling2D()(x)
    
    res  =ResNet50(include_top=False, pooling='avg', weights=None)
    w_layers = _get_weighted_layers(res)
    
    if n>0:
        for layer in res.layers:
            if layer.name in w_layers[-n:]:
                layer.trainable = True
            else:
                layer.trainable=False
    else:
        for layer in res.layers:
            layer.trainable=False
    x = res(inp)
    
    out = Dense(units=25, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

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
#m = resnet_try()
#m = autoencoder(512, 512)
#gram_net = vgg_gram([4])
#m = tan_vgg()
print(m.summary())
#print(m.layers[1].summary())

#m.fit(X, y, epochs=10, batch_size=16)

n_files_train = count_files(data_train)
n_files_test = count_files(data_test)

nepochs = 20
epoch_size = 2500
batch_size = 32
steps_per_epoch = (n_files_train//batch_size)
v_step = 50 #n_files_test//batch_size
distortions=0.1

###
 
#train_datagen = ImageDataGenerator(horizontal_flip =False, preprocessing_function=imagenet_preprocess_input)#, rotation_range=90*distortions,width_shift_range=distortions,height_shift_range=distortions,zoom_range=distortions)
#test_datagen = ImageDataGenerator(preprocessing_function=imagenet_preprocess_input)
#train_generator = train_datagen.flow_from_directory(data_train, target_size=target_size, batch_size=batch_size, class_mode = 'categorical',shuffle=True)
#test_generator = test_datagen.flow_from_directory(data_train, target_size=target_size, batch_size=batch_size, class_mode = 'categorical',shuffle=True)
#
#history=m.fit_generator(train_generator, epochs =nepochs, steps_per_epoch=steps_per_epoch, validation_data = test_generator, validation_steps=v_step)

####

train_paths, y_train, classes = getPaths(data_train)
test_paths, y_test, classes = getPaths(data_test)

prep_func = imagenet_preprocess_input
#prep_func = preprocess_input_vgg

t=time()
X_val, y_val, classes = load_data(data_test, target_size=target_size, prep_func=prep_func)
#X_val, y_val, classes = load_data_part(data_test, target_size=target_size, p=0.5, prep_func=prep_func)
X_tr, y_tr, classes = load_data_part(data_train, target_size=target_size, p=0.05, prep_func=prep_func)
print("Loaded ", time()-t)

#m = load_model('resnet.h5')
#val_loss, val_acc = m.evaluate(X_val, y_val, batch_size)
#print("Val : ", val_loss,  val_acc)

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

###
#from joblib import load
#svms = load('gram_svms.joblib')
#vote_clf = Vote(svms)
#X_val = np.load('precomputed/wiki_vgg_feat_val.npy')
#idxs = [i for i in range(512*512) if i%512<=i//512]
#X_val = np.take(X_val, idxs, axis=-1)
#pred_svm = vote_clf.predict_proba(X_val)
###



#print("SVMS score", categorical_acc(np.argmax(y_val, axis=1), np.argmax(pred_svm, axis=1)))

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


losses=[]
history = {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[], 'tr_acc':[], 'tr_loss':[]}
print('Start training')
for epoch in range(1, nepochs+1):
    print('Epoch ', epoch)
    loss_epoch=[]
    print(q.qsize())
    for batch_idx in tqdm(range(1, steps_per_epoch+1)):
        #print(q.qsize()) 
        
        X_batch, y_batch = q.get()
        loss=m.train_on_batch(x=X_batch, y=y_batch)
        loss_epoch.append(loss)
    print(q.qsize())
    loss_epoch = np.asarray(loss_epoch)
    
    tr_loss = np.mean(loss_epoch[:,0])
    tr_acc = np.mean(loss_epoch[:,1])
    print("Train : ", tr_loss, tr_acc)
    
    val_loss, val_acc = m.evaluate(X_val, y_val, batch_size)
    
    #pred_val = m.predict(X_val,batch_size )
    #val_acc = categorical_acc(np.argmax(y_val, axis=1), np.argmax(pred_val, axis=1))
    
    loss_tr, acc_tr = m.evaluate(X_tr, y_tr, batch_size)
    #acc_tr = evaluate(m, train_paths, y_train, 0.01, target_size=target_size)
    print('Other train : ', loss_tr, acc_tr)
    #m.evaluate(X_val, y_val, batch_size)
    print("Val : ", val_loss,  val_acc)
    
    #mix = pred_val+pred_svm
    #val_mix = categorical_acc(np.argmax(y_val, axis=1), np.argmax(mix, axis=1))
    #print('Mix score :', val_mix)
    
    history['loss'].append(tr_loss)
    history['acc'].append(tr_acc)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    history['tr_loss'].append(loss_tr)
    history['tr_acc'].append(acc_tr)
    
stop_event.set()


#gram_val = gram_net.predict(X_val, batch_size)
#gram_tr = gram_net.predict(X_tr, batch_size)
#
#for epoch in range(1, nepochs+1):
#    print('Epoch ', epoch)
#    loss_epoch=[]
#    print(q.qsize())
#    for batch_idx in tqdm(range(1, steps_per_epoch+1)):
#        #print(q.qsize()) 
#        
#        X_batch, y_batch = q.get()
#        gram_batch = gram_net.predict(X_batch, batch_size)
#        loss=m.train_on_batch(x=gram_batch, y=gram_batch)
#        loss_epoch.append(loss)
#    print(q.qsize())
#    loss_epoch = np.asarray(loss_epoch)
#    
#    tr_loss = np.mean(loss_epoch)
#    print("Train : ", tr_loss)
#    
#    val_loss = m.evaluate(gram_val, gram_val, batch_size)
#    
#    loss_tr = m.evaluate(gram_tr, gram_tr, batch_size)
#    print('Other train : ', loss_tr)
#    print("Val : ", val_loss)
#
#    
#stop_event.set()

#clf = GramPipeline(SGDClassifier(verbose=0, tol=1E-2, n_jobs=4, max_iter=50))
#y_tr_m = np.argmax(y_tr, axis=1)
#y_val_m = np.argmax(y_val, axis=1)

#print('Fitting')
#t=time()
#clf.fit(X_tr, y_tr_m)
#print("Fitted ", time()-t)
#print('Predicting')
#pred = clf.predict(X_val)

#print(categorical_acc(y_val_m, pred))
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB, MultinomialNB
#from sklearn.decomposition import PCA, IncrementalPCA
#from sklearn.pipeline import Pipeline
#from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.externals import joblib
#from sklearn.svm import LinearSVC, SVC, NuSVC
#import sklearn
#
#fractions = 20
#gram_net = vgg_gram([4])
#sgd =  SGDClassifier(verbose=0, tol=1E-2, n_jobs=4, max_iter=20, loss='modified_huber')
#clf = GaussianNB()
#clf = Pipeline([('PCA', PCA(n_components=2048, svd_solver='randomized')), ('CLF', sgd)])
#
#
#names_feat = ['precomputed/wiki_vgg_feat_' + str(i)+'.npy' for i in range(20)]
#names_target = ['precomputed/wiki_vgg_target_' + str(i)+'.npy' for i in range(20)]
#idxs = [i for i in range(512*512) if i%512<=i//512]
#
#pca = PCA(n_components=2048)
#
#k=1
#inc_pca = IncrementalPCA(n_components=2048, copy=False)
#inc_pca = joblib.load('precomputed/incpca2048_k1.joblib')
#for i in range(fractions//k):
#    print("Frac ", i)
#    t=time()
#    X_train = [np.load(names_feat[j]) for j in range(i*k, (i+1)*k)]
#    X_train = np.concatenate(X_train)
#    X_train = np.take(X_train, idxs, axis=-1)
#    inc_pca.partial_fit(X_train)
#    joblib.dump(inc_pca, 'precomputed/incpca2048_k1.joblib')
#    print("Fitted ", time()-t)
    

#X_val = np.load('precomputed/wiki_vgg_feat_val.npy')
#y_val = np.load('precomputed/wiki_vgg_target_val.npy')
#y_val = np.argmax(y_val, axis=1)
#
#X_tr = [np.load(names_feat[j], mmap_mode='r')[::10] for j in range(20)]
#X_tr = np.concatenate(X_tr)
#y_tr = [np.load(names_target[j], mmap_mode='r')[::10] for j in range(20)]
#y_tr = np.concatenate(y_tr)
#y_tr = np.argmax(y_tr, axis=1)
#
#X_val = np.take(X_val, idxs, axis=-1)
#X_tr = np.take(X_tr, idxs, axis=-1)
#
##X_val = inc_pca.transform(X_val)
##X_tr = inc_pca.transform(X_tr)
#X_val = X_val/np.linalg.norm(X_val, axis=0)
#X_tr = X_tr/np.linalg.norm(X_tr, axis=0)
#
#k=1
#svms=[]
#for i in range(fractions//k):
##    if i>1:
#        #clf = RandomForestClassifier(n_estimators=100, n_jobs=16, max_features=256, max_depth=16)
#        clf = SGDClassifier(verbose=0, tol=1E-2, n_jobs=4, max_iter=20, loss='modified_huber')
##        svm = gram_clf(gram_net, clf)
#        print("Frac ", i)
#        t=time()
#        X_train = [np.load(names_feat[j]) for j in range(i*k, (i+1)*k)]
#        X_train = np.concatenate(X_train)
#        X_train = np.take(X_train, idxs, axis=-1)
#        #X_train = inc_pca.transform(X_train)
#        X_train = X_train/np.linalg.norm(X_train, axis=0)
#        
#        y_train = [np.load(names_target[j]) for j in range(i*k, (i+1)*k)]
#        y_train = np.concatenate(y_train)
#        y_train = np.argmax(y_train, axis=1)
#        print("Loaded ", time()-t)
#
#        
#        print('Fitting')
#        t=time()
#        #clf.partial_fit(X_train, y_train_m, classes=np.arange(25), n_iter=1)
#        clf.fit(X_train, y_train)
#        print("Fitted ", time()-t)
#        
#        t=time()
#        print('Predicting')
#        pred_val = clf.predict(X_val[::5])
#        pred_tr = clf.predict(X_tr[::5])
#        print(categorical_acc(y_val[::5], pred_val))
#        print(categorical_acc(y_tr[::5], pred_tr))
#        
#        svms.append(clf)
#        vote_clf = Vote(svms)
#        
#        print('Predicting vote')
#        pred_val, pred_proba_val = vote_clf.predict(X_val[::5])
#        pred_tr, pred_proba_tr = vote_clf.predict(X_tr[::5])
#        print(categorical_acc(y_val[::5], pred_val))
#        print(categorical_acc(y_tr[::5], pred_tr))
#        print('Vote proba')
#        print(categorical_acc(y_val[::5], pred_proba_val))
#        print(categorical_acc(y_tr[::5], pred_proba_tr))
#        print("Tested ", time()-t)
#        
#print('Predicting vote')
#pred_val, pred_proba_val = vote_clf.predict(X_val)
#pred_tr, pred_proba_tr = vote_clf.predict(X_tr)
#print(categorical_acc(y_val, pred_val))
#print(categorical_acc(y_tr, pred_tr))
#print('Vote proba')
#print(categorical_acc(y_val, pred_proba_val))
#print(categorical_acc(y_tr, pred_proba_tr))