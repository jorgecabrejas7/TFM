import numpy as np

def pearsons_r(y_true, y_pred):
    x, y = y_true, y_pred
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x - mx, y - my
    return np.sum(np.multiply(xm,ym)) / np.sqrt(np.multiply(np.sum(np.square(xm)), np.sum(np.square(ym))))

def R2(y_true, y_pred):
    a, b = np.sum(np.square(y_true - y_pred)), np.sum(np.square(y_true - np.mean(y_true)))
    c = a / b
    return 1 - c 

def SMAPE(y_true, y_pred):
    return np.sum(np.abs(y_pred - y_true) / ((y_true + y_pred) / 2)) / len(y_pred)

def MBE(y_true, y_pred):
    return np.mean(y_true - y_pred)
