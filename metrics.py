import keras.backend as K
import tensorflow as tf


def pearsons_r(y_true, y_pred):
        x, y = y_true, y_pred
        mx = K.mean(x)
        my = K.mean(y)
        xm, ym = x - mx, y - my
        return K.sum(tf.multiply(xm,ym)) / K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))

def R2(y_true, y_pred):
    a, b = K.sum(K.square(y_true - y_pred)), K.sum(K.square(y_true - K.mean(y_true)))
    c = a / b
    return 1 - c 

def SMAPE(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true) / ((y_true + y_pred) / 2)) / len(y_pred)

def MBE(y_true, y_pred):
    return K.mean(y_true - y_pred)