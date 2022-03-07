import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.metrics import roc_auc_score


def f1(y_true, y_pred, beta=1):

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    #print('TP :', tp)
    #print('TN :', tn)
    #print("FP :", fp)
    #print('FN :', fn)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    #print('Precision: ', p)
    #print("Recall   : ", r)

    f1 = (1 + beta**2)*p*r / ((beta**2)*p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def accuracy(y_true, y_pred):
    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    accuracy = (tp + tn) / (len(y_pred) + K.epsilon())

    return accuracy

def print_and_save_results(y_true, y_pred, beta=1, save_to='evaluation_results.txt'):

    pred_va_gt_examples = np.hstack((y_true[:], y_pred[:]))
    # calclate AUC befor rounding Y_pred
    auc = roc_auc_score(y_true, y_pred)

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    accuracy = (tp + tn) / (len(y_pred) + K.epsilon())
    sensivity = tp / (tp + fn + K.epsilon())
    specificity = tn / (tn + fp + K.epsilon())

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    ppv = tp / (tp + fp + K.epsilon())
    npv = tn / (tn + fn + K.epsilon())

    f1 = (1 + beta**2)*precision*recall / ((beta**2)*precision+recall+K.epsilon())
    f1 = K.mean(tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1))

    print(f"ACCURACY: {accuracy}\nSENSITIVITY: {sensivity}\nSPECIFICITY: {specificity}\n\nPPV: {ppv}\nNPV: {npv}\n\nPRECISION: {precision}\nRECALL: {recall}\nF1-score: {f1}\n\nAUC: {auc}")

    with open(save_to, 'w') as output_file:
        output_file.write(f"ACCURACY: {accuracy}\nSENSITIVITY: {sensivity}\nSPECIFICITY: {specificity}\n\nPPV: {ppv}\nNPV: {npv}\n\nPRECISION: {precision}\nRECALL: {recall}\nF1-score: {f1}\n\nAUC: {auc}")

    print('Results saved to:', save_to)



def f1_loss(y_true, y_pred, beta=1):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = (1 + beta**2)*p*r / ((beta**2)*p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def f1_multi_class(y_true, y_pred, beta=1):
    n_classes = y_true.shape[1]
    total_f1 = 0
    for c in range(n_classes):
        y_true_binary = y_true[:, c]
        y_pred_binary = y_pred[:, c]
        total_f1 += f1(y_true_binary, y_pred_binary)
    
    return total_f1/n_classes
