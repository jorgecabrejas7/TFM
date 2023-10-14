import tensorflow as tf
from tensorflow.keras import backend as K

def pearsons_r(y_true, y_pred):
    mx = K.mean(y_true)
    my = K.mean(y_pred)
    xm, ym = y_true - mx, y_pred - my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(K.square(xm)) * K.sum(K.square(ym)))
    return r_num / r_den

def r2(mean):
    def R2(y_true, y_pred):
        a = K.sum(K.square(y_true - y_pred))
        b = K.sum(K.square(y_true - mean))
        return 1 - (a / b)
    return R2

def SMAPE(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true) / ((K.abs(y_true) + K.abs(y_pred)) / 2), axis=-1)

def MBE(y_true, y_pred):
    return K.mean(y_true - y_pred)
