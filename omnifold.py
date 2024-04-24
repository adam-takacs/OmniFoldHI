import numpy as np
from matplotlib import pyplot as plt

from keras.layers import Dense, Input
from keras.models import Model

import os
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)

def reweight(events,model,batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights))

def omnifold (theta_gen, theta_sim, theta_dat, model=0, itnum=4, dummy_phi=-10):

    # NN architecture:
    earlystopping = EarlyStopping(patience=10,
                                  verbose=1,
                                  restore_best_weights=True)
    if not model:
        inputs = Input((len(theta_gen[0]), ))
        hidden_layer_1 = Dense(100, activation='relu')(inputs)
        hidden_layer_2 = Dense(100, activation='relu')(hidden_layer_1)
        outputs = Dense(1, activation='sigmoid')(hidden_layer_2)
        model = Model(inputs=inputs, outputs=outputs)

    # deep copy inputs to not change the originals
    theta_gen = theta_gen *1
    theta_sim = theta_sim *1
    theta_dat = theta_dat *1

    # standardize inputs
    X_1 = np.concatenate((theta_sim, theta_dat))
    X_2 = np.concatenate((theta_gen, theta_gen))
    mean_x_1, std_x_1 = (np.mean(X_1[X_1[:,0]!=dummy_phi], axis=0),np.std(X_1[X_1[:,0]!=dummy_phi], axis=0))
    mean_x_2, std_x_2 = (np.mean(X_2[X_2[:,0]!=dummy_phi], axis=0),np.std(X_2[X_2[:,0]!=dummy_phi], axis=0))
    theta_gen[theta_gen[:,0]!=dummy_phi] = (theta_gen[theta_gen[:,0]!=dummy_phi]-mean_x_2)/std_x_2
    theta_sim[theta_sim[:,0]!=dummy_phi] = (theta_sim[theta_sim[:,0]!=dummy_phi]-mean_x_1)/std_x_1
    theta_dat[theta_dat[:,0]!=dummy_phi] = (theta_dat[theta_dat[:,0]!=dummy_phi]-mean_x_1)/std_x_1

    # correct dummy_phi
    theta_gen[theta_gen[:,0]==dummy_phi] = -10
    theta_sim[theta_sim[:,0]==dummy_phi] = -10

    # initial witghts
    weights_pull = np.ones(len(theta_sim))
    weights_push = np.ones(len(theta_gen))

    weights = np.empty(shape=(itnum+1,2,len(theta_sim)))
    weights[0] = (weights_pull, weights_push)

    # omnifold iteration
    for i in range(itnum):

        print("\nITERATION: {}\n".format(i + 1))

        print("STEP 1\n")

        # X_1 -> (sim,dat)
        # Y_1 -> ( 0 , 1 )
        mask = theta_sim[:,0]!=dummy_phi
        xvals_1 = np.concatenate((theta_sim[mask],theta_dat))
        yvals_1 = np.concatenate((np.zeros(len(theta_sim[mask])),np.ones(len(theta_dat))))
        weights_1 = np.concatenate((weights_push[mask], np.ones(len(theta_dat))))

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

        weights_pull = weights_push * reweight(theta_sim,model)

        weights[i+1, 0] = weights_pull
        print('reweighted')

        print("STEP 2\n")

        # X_2 -> (gen,gen)
        # Y_2 -> ( 0 , 1 )
        mask = theta_sim[:,0]!=dummy_phi
        xvals_2 = np.concatenate([theta_gen[mask],theta_gen[mask]])
        yvals_2 = np.concatenate([np.zeros(len(theta_gen[mask])),np.ones(len(theta_gen[mask]))])
        weights_2 = np.concatenate([np.ones(len(theta_gen[mask])),weights_pull[mask]])

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

        weights_push = reweight(theta_gen, model)

        weights[i+1, 1] = weights_push
        print('reweighted')



    return weights




