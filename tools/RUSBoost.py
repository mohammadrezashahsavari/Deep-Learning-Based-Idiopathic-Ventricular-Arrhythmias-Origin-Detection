from genericpath import exists
from pickle import dump
import numpy as np
import pandas as pd
import os
from requests.api import patch
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
import sys

from wfdb.io.record import rdsamp

base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_project_dir)

import utils
from tools.metrics import *


class RUSBoost():
    def __init__(self, X=None, Y=None, n_folds=10, n_classifiers=5, weak_learner=None, min_ration=0.5, sampling_rate=100, aggregation_mode='superdiagnostic', augument_params=(False, None, None), training_parameters_dic={}, base_project_dir='..', output_dir='', task='classification_org'):
        self.X = X
        self.Y = Y
        self.n_folds = n_folds
        self.n_classifiers = n_classifiers
        self.classifier = weak_learner
        self.min_ration = min_ration
        self.sampling_rate = sampling_rate
        self.aggregation_mode = aggregation_mode
        self.augument, self.shifted_seconds, self.steps = augument_params
        self.training_parameters_dic = training_parameters_dic
        self.base_project_dir = base_project_dir
        self.output_dir = output_dir
        self.task = task

        self.classifier.save_weights('initial_weights.h5')

    def perform(self):
        kfold = KFold(self.n_folds)
        output_models_dir = os.path.join(self.base_project_dir, 'output_models')
        if not os.path.exists(output_models_dir):
            os.mkdir(output_models_dir)
        output_models_dir = os.path.join(output_models_dir, self.task)
        if not os.path.exists(output_models_dir):
            os.mkdir(output_models_dir)
        Y_test_set = list()
        Y_pred_set = list()
        for i, (train_val, test) in enumerate(kfold.split(self.Y)):
            X_train_val, self.X_test = self.X[train_val], self.X[test]
            Y_train_val, self.Y_test = self.Y[train_val], self.Y[test]    
            self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.12, random_state=1)

            print('\n\n')    
            self.X_train, self.X_val, ss = utils.preprocess_signals(self.X_train, self.X_val, True)
            self.X_test = utils.apply_standardizer(self.X_test, ss)

            if self.augument:
                self.X_train, self.Y_train = utils.augment_signals(self.X_train, self.Y_train, self.shifted_seconds, self.steps, self.sampling_rate)
                total_shifts = int(self.shifted_seconds*self.steps*self.sampling_rate)
                self.X_val = self.X_val[:, total_shifts:, :]
                self.X_test = self.X_test[:, total_shifts:, :]

            self.output_models_fold_dir = os.path.join(output_models_dir, 'fold' + str(i))
            if not os.path.exists(self.output_models_fold_dir):
                os.mkdir(self.output_models_fold_dir)
            self.classifiers_weight = np.zeros((self.n_classifiers, 1))
            for c in range(self.n_classifiers):
                print('Training {0}th model of {1}th Fold.'.format(c+1, i+1))
                self.classifier.load_weights("initial_weights.h5")
                n_samples = self.Y_train.shape[0]
                self.sample_weights = (1/n_samples) * np.ones((n_samples, 1))
                rused_X, rused_Y, rused_weighs = self.random_undersampling()

                self.classifier.fit(
                    rused_X,
                    rused_Y,
                    sample_weight=rused_weighs,
                    validation_data=(self.X_val, self.Y_val),
                    **self.training_parameters_dic
                )

                Y_pred = self.classifier.predict(self.X_train)
                Y_pred = np.round(Y_pred)

                false_preds = 1 - (Y_pred==self.Y_train).astype(np.int32)

                loss = np.sum(self.sample_weights*false_preds)

                self.classifiers_weight[c] = loss/(1-loss)

                for s in range(self.sample_weights.shape[0]):
                    if self.Y_train[s] != Y_pred[s]:
                        self.sample_weights[s] *= loss/(1-loss)
                
                self.sample_weights /= np.sum(self.sample_weights)

                output_moedel_path = os.path.join(self.output_models_fold_dir, self.task + str(c) + '.h5')
                self.classifier.save_weights(output_moedel_path)

            self.classifiers_weight = np.log(1/self.classifiers_weight)
            self.classifiers_weight /= np.sum(self.classifiers_weight)

            Y_test_set.append(self.Y_test)
            Y_pred_set.append(self.predict(self.X_test))

            classifiers_weights_path = os.path.join(self.output_models_fold_dir, 'classifiers_weight.pickle')
            with open(classifiers_weights_path, 'wb') as f:
                dump(self.classifiers_weight, f)
            
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.evaluate(Y_pred_set, Y_test_set)
        os.remove("initial_weights.h5")

        '''classifiers_file_path = os.path.join(self.output_dir, 'classifiers_sets.pickle')
        with open(classifiers_file_path, 'wb') as f:
            dump(classifiers_sets, f)'''
        

    def predict(self, X):
        Y_pred = np.zeros((X.shape[0], 1))
        for c in range(self.n_classifiers):
            moedel_path = os.path.join(self.output_models_fold_dir, self.task + str(c) + '.h5')
            self.classifier.load_weights(moedel_path)
            Y_pred += (self.classifier.predict(X))*self.classifiers_weight[c]

        return Y_pred

    def evaluate(self, Y_pred_set, Y_test_set):
        Y_pred_total = Y_pred_set[0]
        Y_total = Y_test_set[0]
        for i in range(1, len(Y_pred_set)):
            Y_pred_total = np.vstack((Y_pred_total, Y_pred_set[i]))
            Y_total = np.vstack((Y_total, Y_test_set[i]))

        save_results_to = os.path.join(self.output_dir, self.task + '.txt')
        print_and_save_results(Y_total, Y_pred_total, save_to=save_results_to)

    def random_undersampling(self):
        n_samples = self.Y_train.shape[0]

        # DEBUGING STARTS
        idxes = np.zeros((n_samples, 1))
        for i in range(n_samples):
            idxes[i] = i
        # DEBUGING ENDS

        maj_class = 0
        if np.sum(self.Y_train) > n_samples/2:
            n_maj = np.sum(self.Y_train)
            n_min = n_samples - n_maj
            maj_class = 1
        else:
            n_min = np.sum(self.Y_train)
            n_maj = n_samples - n_min
            maj_class = 0

        maj_idxs = (self.Y_train == maj_class).reshape(-1)
        maj_Y = self.Y_train[maj_idxs].reshape(-1, 1)
        maj_X = self.X_train[maj_idxs]
        maj_weights = self.sample_weights[maj_idxs].reshape(-1, 1)
        maj_idxes = idxes[maj_idxs].reshape(-1, 1)                              #DEBUGING

        min_idxs = (self.Y_train == (1-maj_class)).reshape(-1)
        min_Y = self.Y_train[min_idxs].reshape(-1, 1)
        min_X = self.X_train[min_idxs]
        min_weights = self.sample_weights[min_idxs].reshape(-1, 1)
        min_idxes = idxes[min_idxs].reshape(-1, 1)                              #DEBUGING

        # alternative reperesentation of "n_min/(n_maj_remaining + n_min) = self.min_ration" equation
        n_maj_remaining = int((n_min*(1 - self.min_ration)/self.min_ration))

        permutation = np.random.permutation(maj_Y.shape[0])
        permutation = np.random.choice(permutation, n_maj_remaining, False)
        maj_remainin_X = maj_X[permutation]
        maj_remainin_Y = maj_Y[permutation].reshape(-1, 1)
        maj_remainin_weights = maj_weights[permutation].reshape(-1, 1)
        maj_remaining_idxes = maj_idxes[permutation].reshape(-1, 1)             #DEBUGING

        rused_X = np.vstack((maj_remainin_X, min_X))
        rused_Y = np.vstack((maj_remainin_Y, min_Y))
        rused_weights = np.vstack((maj_remainin_weights, min_weights))
        rused_idxes = np.vstack((maj_remaining_idxes, min_idxes))               #DEBUGING


        permutation = np.random.permutation(rused_Y.shape[0])
        rused_X = rused_X[permutation]
        rused_Y = rused_Y[permutation].reshape(-1, 1)
        rused_weights = rused_weights[permutation].reshape(-1, 1)
        rused_idxes = rused_idxes[permutation].reshape(-1, 1)                   #DEBUGING

        a = rused_idxes[:2].reshape(-1).astype(np.int32)                        #DEBUGING

        #print(rused_X[:2], rused_Y[:2], self.X[a], self.Y[a], a, sep='\n\n')   #DEBUGING

        return rused_X, rused_Y, rused_weights
        





