from copy import deepcopy
import os
from numpy.core.defchararray import index
from numpy.core.fromnumeric import reshape
import pandas as pd
import numpy as np
from scipy.sparse import data
from sklearn import preprocessing
from scipy.io import loadmat
from tensorflow.keras import layers
from tqdm import tqdm
from random import uniform
from librosa.core import resample
import matplotlib.pyplot as plt

base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def load_database(database_path):
    print('Loading \'Diagnosis.xlsx\' ...')
    database = pd.read_csv(database_path, index_col='HospitalID')
    return database

def convert_encoded_lables_to_one_hot(encoded_lables, lable_encoder):
    n_calsses = len(lable_encoder.classes_)
    n_subjects = len(encoded_lables)
    one_hot_lables = np.zeros((n_subjects, n_calsses))
    for i in range(n_subjects):
        subject_encoded_label = encoded_lables[i]
        one_hot_lables[i, subject_encoded_label] = 1
    return one_hot_lables

def preprocess_lables(database, diag_mode='LeftRight'):
    database.dropna(subset = [diag_mode], inplace=True)
    le = preprocessing.LabelEncoder()
    le.fit(list(database[diag_mode]))
    lables = le.transform(list(database[diag_mode]))
    print('Classes:', le.classes_)
    if len(le.classes_) > 2:
        lables = convert_encoded_lables_to_one_hot(lables, le)
    else:
        #lables = [0 if lable == 1 else 0 for lable in lables]
        lables = np.array(lables).reshape(-1, 1)

    return lables, database, le


def load_raw_signals(dataset_path, database):
    ecg_signals = np.array([loadmat(os.path.join(dataset_path, str(index) + '.mat')).get('filtered_signal_50hz') for index in tqdm(database.index)])  #  padded_ecg_signal
    return ecg_signals

def preprocess_signals(training_ecg_signals, testing_ecg_signals, return_ss=False):
    # Standardize data such that mean 0 and variance 1
    print('Standardizing data set ...')
    ss = preprocessing.StandardScaler()
    ss.fit(np.vstack(training_ecg_signals).flatten()[:,np.newaxis].astype(float))

    if return_ss:
        return apply_standardizer(training_ecg_signals, ss), apply_standardizer(testing_ecg_signals, ss), ss
    else:
        return apply_standardizer(training_ecg_signals, ss), apply_standardizer(testing_ecg_signals, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp

def reverse_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.inverse_transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp

def normalize(tSignal):
      # copy the data if needed, omit and rename function argument if desired
      signal = np.copy(tSignal) # signal is in range [a;b]
      signal -= np.min(signal) # signal is in range to [0;b-a]
      signal /= np.max(signal) # signal is normalized to [0;1]
      signal -= 0.5 # signal is in range [-0.5;0.5]
      signal *=2 # signal is in range [-1;1]
      return signal

def augment_signals_by_shifting(ecg_signals, binary_labels, shifted_seconds=0.2, steps=5, sampling_rate=50, seed=0):
    print(f'Augmenting data by shifting. shifted_seconds:{shifted_seconds} step:{steps}  ...')
    n_samples = ecg_signals.shape[1]
    total_shifts = int(shifted_seconds*steps*sampling_rate)
    n_signals = ecg_signals.shape[0]
    augmented_ecg_signals = ecg_signals[:, total_shifts:, :]
    for i in tqdm(range(n_signals)):
        ecg_signal = ecg_signals[i]
        for step in range(1, steps+1):
            #print('n_samples:', n_samples)
            n_shifts = int(shifted_seconds*step*sampling_rate)
            augmented_ecg_signal = np.zeros((n_samples, 12))
            augmented_ecg_signal[n_shifts: ,:] = ecg_signal[:n_samples-n_shifts, :]
            
            #DEBUGING STARTS  
            '''if n_shifts==20:
                plt.figure()
                time_axies = list(map(lambda x:x/sampling_rate, [i for i in range(n_samples)]))
                plt.plot(time_axies, ecg_signal[:, 0], label='Orginal ECG')
                plt.plot(time_axies, augmented_ecg_signal[:, 0], label='Augmented ECG with {shift}s shift'.format(shift=n_shifts/sampling_rate))
                plt.xlabel('Time(s)')
                plt.ylabel('ECG signal')
                plt.legend()
                plt.show()'''
            #DEBUGING ENDS

            augmented_ecg_signal = augmented_ecg_signal[total_shifts:, :]
            augmented_ecg_signal = np.expand_dims(augmented_ecg_signal, axis=0)
            augmented_ecg_signals = np.vstack((augmented_ecg_signals, augmented_ecg_signal))

        augmented_labels = binary_labels[i] * np.ones((steps, 1))
        binary_labels = np.vstack((binary_labels, augmented_labels))
    
    ecg_signals = ecg_signals[:, steps*n_shifts:, :]

    #DEBUGING STARTS
    '''while True:
        x = input('Enter an index:')
        if x == 'q':
            break
        x = int(x)
        print(binary_labels[x], binary_labels[88 + steps*x: 88+steps*(x+1)])'''
    #DEBUGING ENDS
    np.random.seed(seed)
    permutation = np.random.permutation(augmented_ecg_signals.shape[0])
    augmented_ecg_signals = augmented_ecg_signals[permutation]
    binary_labels = binary_labels[permutation].reshape(-1, 1)
        
    return augmented_ecg_signals, binary_labels


def augment_signals_by_rescaling(ecg_signals, labels, rescale_range=[0.8, 1.2], n_out_of_each=50, sampling_rate=50, seed=0):
    print('Augmenting data by rescaling ...')
    n_signals = ecg_signals.shape[0]
    n_classes = labels.shape[1]
    max_ratio = rescale_range[1]
    org_n_samples = ecg_signals.shape[1]
    augmented_ecg_signals = [zero_padd_signal(ecg_signals[i], int(max_ratio*org_n_samples), 12) for i in range(n_signals)]
    for i in tqdm(range(n_signals)):
        ecg_signal = ecg_signals[i]
        for _ in range(n_out_of_each):
            rate = uniform(*rescale_range)
            rescaled_n_samples = int(rate*org_n_samples)
            augmented_ecg_signal = np.zeros((int(max_ratio*org_n_samples), 12))
            for c in range(12):
                augmented_ecg_signal[:, c] = random_rescale_and_zeropadd_signal(ecg_signal[:, c], org_n_samples, rescaled_n_samples, rescale_range[1], sampling_rate)
            #DEBUGING STARTS  
            '''augmented_n_sampled = augmented_ecg_signal.shape[0]
            time_axies_org = list(map(lambda x:x/sampling_rate, [i for i in range(org_n_samples)]))
            time_axies_augmented = list(map(lambda x:x/sampling_rate, [i for i in range(augmented_n_sampled)]))
            plt.xlabel('Time(s)')
            plt.ylabel('Padded ECG signal')
            plt.plot(time_axies_org, ecg_signal[:, 0], time_axies_augmented, augmented_ecg_signal[:, 0])
            plt.legend(['Org signal', 'Augmented signal with:%.2f factor'%rate])
            plt.show()
            continue'''
            #DEBUGING ENDS

            augmented_ecg_signals.append(augmented_ecg_signal)

        augmented_labels = labels[i] * np.ones((n_out_of_each, n_classes))
        labels = np.vstack((labels, augmented_labels))

    #DEBUGING STARTS
    '''while True:
        x = input('Enter an index:')
        if x == 'q':
            break
        x = int(x)
        print(labels[x], labels[ecg_signals.shape[0] + n_out_of_each*x: ecg_signals.shape[0]+n_out_of_each*(x+1)])'''
    #DEBUGING ENDS

    augmented_ecg_signals = np.array(augmented_ecg_signals)
    labels = np.array(labels).reshape(-1, 1)
    np.random.seed(seed)
    permutation = np.random.permutation(augmented_ecg_signals.shape[0])
    augmented_ecg_signals = augmented_ecg_signals[permutation]
    labels = labels[permutation].reshape(-1, 1)
    
    return augmented_ecg_signals, labels


def random_rescale_and_zeropadd_signal(ecg_signal, from_sample, to_samples, max_ratio=1.2, samplig_rate=50):
    augmented_ecg_signal = resample(ecg_signal, from_sample, to_samples)[:to_samples]
    padded_n_sampled = int(from_sample * max_ratio)
    augmented_ecg_signal = zero_padd_signal(augmented_ecg_signal, padded_n_sampled, 1)
    return augmented_ecg_signal

def self_padd_signal(ecg_signal, padded_n_sampled, n_channels=1):
    if n_channels == 1:
        org_n_samples = ecg_signal.shape[0]
        padded_ecg_signal = np.zeros((padded_n_sampled, ))
        for i in range(int(np.floor([padded_n_sampled/org_n_samples]))):
            padded_ecg_signal[org_n_samples*i: org_n_samples*(i+1)] = ecg_signal

        start_idx = int(np.floor([padded_n_sampled/org_n_samples])) * org_n_samples
        padded_ecg_signal[start_idx:] = ecg_signal[:padded_n_sampled - start_idx]
        return padded_ecg_signal
    else:
        org_n_samples = ecg_signal.shape[0]
        padded_ecg_signal = np.zeros((padded_n_sampled, n_channels))
        for i in range(int(np.floor([padded_n_sampled/org_n_samples]))):
            padded_ecg_signal[org_n_samples*i: org_n_samples*(i+1), :] = ecg_signal

        start_idx = int(np.floor([padded_n_sampled/org_n_samples])) * org_n_samples
        padded_ecg_signal[start_idx:] = ecg_signal[:padded_n_sampled - start_idx, :]
        return padded_ecg_signal

def zero_padd_signal(ecg_signal, padded_n_sampled, n_channels=1):
    if n_channels == 1:
        org_n_samples = ecg_signal.shape[0]
        padded_ecg_signal = np.zeros((padded_n_sampled, ))
        padded_ecg_signal[:org_n_samples] = ecg_signal
        return padded_ecg_signal
    else:
        org_n_samples = ecg_signal.shape[0]
        padded_ecg_signal = np.zeros((padded_n_sampled, n_channels))
        padded_ecg_signal[:org_n_samples, :] = ecg_signal
        return padded_ecg_signal


def crop_test_signals(ecg_signals, shifted_seconds, steps, croped_shape, sampling_rate=50):
    n_shifts = int(sampling_rate*shifted_seconds*steps)
    croped_ecg_signals = np.zeros((ecg_signals.shape[0], *croped_shape))
    for i in range(ecg_signals.shape[0]):
        croped_ecg_signals[i] = ecg_signals[i, :-n_shifts, :]

    return croped_ecg_signals

def self_padd_test_signals(ecg_signals, rescale_range=[0.8, 1.2]):
    from_sample = ecg_signals.shape[1]
    padded_ecg_signals = [self_padd_signal(ecg_signals[i], int(from_sample*rescale_range[1]), 12) for i in range(ecg_signals.shape[0])]
    return np.array(padded_ecg_signals)






class Dataset10FoldSpliter():
    def __init__(self, X, Y, shuffle=False, seed=0):
        self.X = X
        self.Y = Y
        self.shuffle = shuffle
        self.seed = seed

        batch_size = X.shape[0]
        fold_size = batch_size // 10
        #creating fold numbers for each ECG
        self.fold_numbers = np.arange(1, 11)
        self.fold_numbers = np.repeat(self.fold_numbers, fold_size)
        for _ in range(batch_size - fold_size*10):
            self.fold_numbers = np.append(self.fold_numbers, 10) 

        self.test_trackers = np.arange(batch_size)
        if shuffle:
            print(f'Shuffling dataset with seed: {seed}.')
            np.random.seed(seed)
            permutation = np.random.permutation(batch_size)
            self.X = self.X[permutation]
            self.Y = self.Y[permutation].reshape(-1, 1)
            self.test_trackers = self.test_trackers[permutation].reshape(-1, 1)
            
        self.val_fold = 1
        self.test_fold = 2
    
    def split(self):
        not_val_idxs = self.fold_numbers != self.val_fold
        not_test_idxs = self.fold_numbers != self.test_fold
        train_idxs = not_val_idxs * not_test_idxs

        val_idxs = self.fold_numbers == self.val_fold
        test_idxs = self.fold_numbers == self.test_fold

        X_train = self.X[train_idxs]
        Y_train = self.Y[train_idxs].reshape(-1, 1)
        X_val = self.X[val_idxs]
        Y_val = self.Y[val_idxs].reshape(-1, 1)
        X_test = self.X[test_idxs]
        Y_test = self.Y[test_idxs].reshape(-1, 1)
        test_fold_tracker = self.test_trackers[test_idxs].reshape(-1, 1)
        
        self.val_fold += 1
        self.test_fold = self.test_fold+1 if self.val_fold != 10 else 1

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, test_fold_tracker
        



if __name__ == '__main__':
    pass
    
    