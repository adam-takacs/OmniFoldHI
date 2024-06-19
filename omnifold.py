import numpy as np
from matplotlib import pyplot as plt

from keras.layers import Dense, Input
from keras.models import Model

import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # AF: makes sure there are no explosions

    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)

def weighted_binary_crossentropy_square(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    t_loss = -weights**2 * ((y_true) * K.log(y_pred) +
                            (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)

def reweight(events,model,batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights))

def omnifold_weights (events_gen, events_sim, events_dat, weights_syn=[], weights_nat=[], iterations=4, model=0, dummy_phi=-10, rand_seed=239):

    # Initialization seed
    random.seed(rand_seed)
    np.random.seed(rand_seed+934)
    tf.random.set_seed(rand_seed+534)

    # NN architecture:
    earlystopping = EarlyStopping(patience=10,
                                  verbose=1,
                                  restore_best_weights=True)
    if not model:
        inputs = Input((len(events_gen[0]), ))
        hidden_layer_1 = Dense(100, activation='relu')(inputs)
        hidden_layer_2 = Dense(100, activation='relu')(hidden_layer_1)
        outputs = Dense(1, activation='sigmoid')(hidden_layer_2)
        model = Model(inputs=inputs, outputs=outputs)

    # deep copy inputs to not change the originals
    events_gen = events_gen *1
    events_sim = events_sim *1
    events_dat = events_dat *1

    # standardize inputs
    X_1 = np.concatenate((events_sim, events_dat))
    X_2 = np.concatenate((events_gen, events_gen))
    mean_x_1, std_x_1 = (np.mean(X_1[X_1[:,0]!=dummy_phi], axis=0),np.std(X_1[X_1[:,0]!=dummy_phi], axis=0))
    mean_x_2, std_x_2 = (np.mean(X_2[X_2[:,0]!=dummy_phi], axis=0),np.std(X_2[X_2[:,0]!=dummy_phi], axis=0))
    events_gen[events_gen[:,0]!=dummy_phi] = (events_gen[events_gen[:,0]!=dummy_phi]-mean_x_2)/std_x_2
    events_sim[events_sim[:,0]!=dummy_phi] = (events_sim[events_sim[:,0]!=dummy_phi]-mean_x_1)/std_x_1
    events_dat[events_dat[:,0]!=dummy_phi] = (events_dat[events_dat[:,0]!=dummy_phi]-mean_x_1)/std_x_1

    # correct dummy_phi
    events_gen[events_gen[:,0]==dummy_phi] = -10
    events_sim[events_sim[:,0]==dummy_phi] = -10

    # initial weights
    if len(weights_syn):
        weights_pull = weights_syn *1
        weights_push = weights_syn *1
    else:
        weights_pull = np.ones(len(events_sim))
        weights_push = np.ones(len(events_gen))

    # data weights
    if len(weights_nat):
        weights_dat = weights_nat *1
    else:
        weights_dat = np.ones(len(events_dat))


    weights = np.empty(shape=(iterations+1,2,len(events_sim)))
    weights[0] = (weights_pull, weights_push)

    # _weights iteration
    for i in range(iterations):

        print("\nITERATION: {}\n".format(i + 1))

        print("STEP 1\n")

        # X_1 -> (sim,dat)
        # Y_1 -> ( 0 , 1 )
        mask = events_sim[:,0] != -10
        xvals_1 = np.concatenate((events_sim[mask],events_dat))
        yvals_1 = np.concatenate((np.zeros(len(events_sim[mask])),np.ones(len(events_dat))))
        weights_1 = np.concatenate((weights_push[mask], weights_dat))

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1)

        # zip ("hide") the weights with the labels
        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)

        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=['accuracy'],
                      weighted_metrics=[])
        print('compiled')
        model.fit(X_train_1,
                  Y_train_1,
                  epochs=200,
                  batch_size=10000,
                  validation_data=(X_test_1, Y_test_1),
                  callbacks=[earlystopping],
                  verbose=1)
        print('fitted')

        weights_pull = reweight(events_sim,model)

        weights[i+1, 0] = weights_pull
        print('reweighted')

        print("STEP 2\n")

        # X_2 -> (gen,gen)
        # Y_2 -> ( 0 , 1 )
        mask = events_sim[:,0] != -10
        xvals_2 = np.concatenate([events_gen[mask],events_gen[mask]])
        yvals_2 = np.concatenate([np.zeros(len(events_gen[mask])),np.ones(len(events_gen[mask]))])
        weights_2 = np.concatenate([np.ones(len(events_gen[mask])),weights_pull[mask]*weights_push[mask]])

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2)

        # zip ("hide") the weights with the labels
        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)
        
        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=['accuracy'],
                      weighted_metrics=[])
        model.fit(X_train_2,
                  Y_train_2,
                  epochs=200,
                  batch_size=10000,
                  validation_data=(X_test_2, Y_test_2),
                  callbacks=[earlystopping],
                  verbose=1)

        weights_push = reweight(events_gen, model)

        weights[i+1, 1] = weights_push
        print('reweighted')

    return weights

def omnifold_weights2 (events_gen, events_sim, events_dat, weights_syn=[], weights_nat=[], iterations=4, model=0, dummy_phi=-10, rand_seed=239):

    # Initialization seed
    random.seed(rand_seed)
    np.random.seed(rand_seed+934)
    tf.random.set_seed(rand_seed+534)

    # NN architecture:
    earlystopping = EarlyStopping(patience=10,
                                  verbose=1,
                                  restore_best_weights=True)
    if not model:
        inputs = Input((len(events_gen[0]), ))
        hidden_layer_1 = Dense(100, activation='relu')(inputs)
        hidden_layer_2 = Dense(100, activation='relu')(hidden_layer_1)
        outputs = Dense(1, activation='sigmoid')(hidden_layer_2)
        model = Model(inputs=inputs, outputs=outputs)

    # deep copy inputs to not change the originals
    events_gen = events_gen *1
    events_sim = events_sim *1
    events_dat = events_dat *1

    # standardize inputs
    X_1 = np.concatenate((events_sim, events_dat))
    X_2 = np.concatenate((events_gen, events_gen))
    mean_x_1, std_x_1 = (np.mean(X_1[X_1[:,0]!=dummy_phi], axis=0),np.std(X_1[X_1[:,0]!=dummy_phi], axis=0))
    mean_x_2, std_x_2 = (np.mean(X_2[X_2[:,0]!=dummy_phi], axis=0),np.std(X_2[X_2[:,0]!=dummy_phi], axis=0))
    events_gen[events_gen[:,0]!=dummy_phi] = (events_gen[events_gen[:,0]!=dummy_phi]-mean_x_2)/std_x_2
    events_sim[events_sim[:,0]!=dummy_phi] = (events_sim[events_sim[:,0]!=dummy_phi]-mean_x_1)/std_x_1
    events_dat[events_dat[:,0]!=dummy_phi] = (events_dat[events_dat[:,0]!=dummy_phi]-mean_x_1)/std_x_1

    # correct dummy_phi
    events_gen[events_gen[:,0]==dummy_phi] = -10
    events_sim[events_sim[:,0]==dummy_phi] = -10

    # initial weights
    if len(weights_syn):
        weights_pull = weights_syn *1
        weights_push = weights_syn *1
    else:
        weights_pull = np.ones(len(events_sim))
        weights_push = np.ones(len(events_gen))

    # data weights
    if len(weights_nat):
        weights_dat = weights_nat *1
    else:
        weights_dat = np.ones(len(events_dat))


    weights = np.empty(shape=(iterations+1,2,len(events_sim)))
    weights[0] = (weights_pull, weights_push)

    # _weights iteration
    for i in range(iterations):

        print("\nITERATION: {}\n".format(i + 1))

        print("STEP 1\n")

        # X_1 -> (sim,dat)
        # Y_1 -> ( 0 , 1 )
        mask = events_sim[:,0] != -10
        xvals_1 = np.concatenate((events_sim[mask],events_dat))
        yvals_1 = np.concatenate((np.zeros(len(events_sim[mask])),np.ones(len(events_dat))))
        weights_1 = np.concatenate((weights_push[mask], weights_dat))

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1)

        # zip ("hide") the weights with the labels
        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)

        model.compile(loss=weighted_binary_crossentropy_square,
                      optimizer='Adam',
                      metrics=['accuracy'],
                      weighted_metrics=[])
        print('compiled')
        model.fit(X_train_1,
                  Y_train_1,
                  epochs=200,
                  batch_size=10000,
                  validation_data=(X_test_1, Y_test_1),
                  callbacks=[earlystopping],
                  verbose=1)
        print('fitted')

        weights_pull = reweight(events_sim,model)**.5

        weights[i+1, 0] = weights_pull
        print('reweighted')

        print("STEP 2\n")

        # X_2 -> (gen,gen)
        # Y_2 -> ( 0 , 1 )
        mask = events_sim[:,0] != -10
        xvals_2 = np.concatenate([events_gen[mask],events_gen[mask]])
        yvals_2 = np.concatenate([np.zeros(len(events_gen[mask])),np.ones(len(events_gen[mask]))])
        weights_2 = np.concatenate([np.ones(len(events_gen[mask])),weights_pull[mask]*weights_push[mask]])

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2)

        # zip ("hide") the weights with the labels
        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)
        
        model.compile(loss=weighted_binary_crossentropy_square,
                      optimizer='Adam',
                      metrics=['accuracy'],
                      weighted_metrics=[])
        model.fit(X_train_2,
                  Y_train_2,
                  epochs=200,
                  batch_size=10000,
                  validation_data=(X_test_2, Y_test_2),
                  callbacks=[earlystopping],
                  verbose=1)

        weights_push = reweight(events_gen, model)**.5

        weights[i+1, 1] = weights_push
        print('reweighted')

    return weights




