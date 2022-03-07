import os
from types import new_class
import utils
import numpy as np
from Models.SemisupervisedModels import *
from tools.metrics import *
from tools.losses import *
from tools import callbacks
from random import shuffle
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from librosa.core import resample
import pickle

class SemiSupervisedExperiment():
    def __init__(self, dataset_path, database_path=None, sampleing_rate=50, diag_mode='LeftRight', n_folds=10, base_project_dir='.', seed=0, augmentition=None, network_structure='VGG_LSTM_Attn', start_fold=1, use_pre_trained=False, plot_attention_weights=False):
        self.dataset_path = dataset_path
        self.database_path = database_path
        self.sampleing_rate = sampleing_rate
        self.diag_mode = diag_mode
        self.n_folds = n_folds
        self.base_project_dir = base_project_dir
        self.seed = seed
        self.augmentition = augmentition
        if augmentition:
            self.augmentition_type = augmentition['type']
            self.augmentition_params = augmentition['params']
        else:
            self.augmentition_type = None
            self.augmentition_params = None
        self.network_structure = network_structure
        self.start_fold = start_fold
        self. task = str()
        self.use_pre_trained = use_pre_trained
        self.plot_attention_weights = plot_attention_weights

        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        tf.random.set_seed(seed)

    def prepare(self):
        if not self.database_path:
            self.database_path = os.path.join(self.dataset_path, '..', 'Diagnosis.csv')

        self.database = utils.load_database(self.database_path)
        self.database = self.database[:]
        self.Y, database, self.lable_encoder = utils.preprocess_lables(self.database, self.diag_mode)
        self.X = utils.load_raw_signals(self.dataset_path, database)
        #self.X = self.X[:, :1200, :]

        with open('signals_with_PVC.npy', 'rb') as f:
            self.unlabled_X = np.load(f)

        zero_padded_unabed_X = np.zeros((self.unlabled_X.shape[0], self.X.shape[1], self.X.shape[2]))
        for i in range(self.unlabled_X.shape[0]):
            zero_padded_unabed_X[i] = utils.zero_padd_signal(self.unlabled_X[i], self.X.shape[1], n_channels=12)
        
        self.unlabeled_X = zero_padded_unabed_X

        print('Shuffling dataset.')
        np.random.seed(self.seed)
        permutation = np.random.permutation(self.X.shape[0])
        self.X = self.X[permutation]
        self.Y = self.Y[permutation].reshape(-1, 1)

        np.random.seed(self.seed)
        permutation = np.random.permutation(self.unlabeled_X.shape[0])
        self.unlabeled_X = self.unlabeled_X[permutation]

        self.n_classes = len(self.lable_encoder.classes_)
        if self.augmentition_type == 'shifting':
            self.input_shape = (int(self.X.shape[1] - self.sampleing_rate*self.augmentition_params[0]*self.augmentition_params[1]), self.X.shape[2])
            self.task = self.task + 'ShifAug'
        elif  self.augmentition_type == 'rescaling':
            self.input_shape = (int(self.augmentition_params[0][1] * self.X.shape[1]), self.X.shape[2])
            self.task = self.task + 'RescaleAug'
        else:
            self.input_shape = self.X.shape[1:]
            self.task = self.task + 'Org'
            
        print('Labled Input Shape:', self.input_shape)
        print('Unlabled Input Shape:', self.unlabeled_X.shape)
        print('Number of Classes:', self.n_classes)
        print('Number of Positive Classes:', np.sum(self.Y))
        
    def train(self):
        encoder = EncoderModel(self.input_shape)
        encoder.summary()
        print('\n\n')
        decoder = DecoderModel(encoder, output_shape=self.input_shape)
        decoder.summary()
        print('\n\n')
        classifier = classifierModel(encoder, 1)
        classifier.summary()
        print('\n\n')
        
        model = SemiSupervisedModel(encoder, decoder, classifier)
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, clipvalue=5.0),
            en_de_loss_fn = tf.keras.losses.MSE,
            classifier_loss_fn = BinaryCrossEntropy,
            metric = f1,
            n_mini_batches = 10,
            unsupervised_weight=10
        )

        self.X_train_val, self.X_test, self.Y_train_val, self.Y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=self.seed)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train_val, self.Y_train_val, test_size=0.12, random_state=self.seed)

        print('\n\n')    
        n_X_train = self.X_train.shape[0]
        temp_X = np.vstack((self.X_train, self.unlabeled_X))
        temp_X, self.X_val, ss = utils.preprocess_signals(temp_X, self.X_val, True)
        self.X_test = utils.apply_standardizer(self.X_test, ss)

        self.X_train = temp_X[:n_X_train]
        self.unlabeled_X = temp_X[n_X_train:]

        data = {
            'labeled_signals':self.X_train,
            'unlabeled_signals':self.unlabeled_X,
            'train_labels':self.Y_train,
            'val_signals':self.X_val,
            'val_labels':self.Y_val
        }

        try:
            model.fit(
                data,
                epochs=100,
            )
        except KeyboardInterrupt:
            pass

        test_encoder_output = model.encoder(self.X_test)
        self.task += '_SemiSupervised'
        self.output_dir = os.path.join(self.base_project_dir, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        save_results_to = os.path.join(self.output_dir, self.task + '.txt')
        print_and_save_results(self.Y_test, model.classifier(test_encoder_output), save_to=save_results_to)

        self.TrainedModels_dir = os.path.join(self.base_project_dir, 'TrainedModels')
        if not os.path.exists(self.TrainedModels_dir):
            os.mkdir(self.TrainedModels_dir)
        encoder_path = os.path.join(self.TrainedModels_dir, 'encoder.h5')
        decoder_path = os.path.join(self.TrainedModels_dir, 'decoder.h5')
        classifier_path = os.path.join(self.TrainedModels_dir, 'classifier.h5')
        
        model.encoder.save_weights(encoder_path)
        model.decoder.save_weights(decoder_path)
        model.classifier.save_weights(classifier_path)
    
        batch_size = 100
        
        reconstructed_signals = model.decoder(test_encoder_output[0])
        n_samples = self.X_test.shape[1]
        use_lead = 0
        for i in range(batch_size):
            time_axies = list(map(lambda x:x/self.sampleing_rate, [i for i in range(n_samples)]))
            plt.figure()
            plt.title('Reconstructed ECG')
            plt.xlabel('Time(s)')
            plt.ylabel('Reconstructed ECG signal')
            plt.plot(time_axies, reconstructed_signals[i, :, use_lead])
            plt.figure()
            plt.title('Original ECG')
            plt.xlabel('Time(s)')
            plt.ylabel('Real ECG signal')
            plt.plot(time_axies, self.X_test[i, :, use_lead])
            plt.show()
            input()


    def train_10fold(self):
        encoder = EncoderModel(self.input_shape)
        encoder.summary()
        print('\n\n')
        decoder = DecoderModel(encoder, output_shape=self.input_shape)
        decoder.summary()
        print('\n\n')
        classifier = classifierModel(encoder, 1)
        classifier.summary()
        print('\n\n')
        
        model = SemiSupervisedModel(encoder, decoder, classifier)
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=5.0),
            en_de_loss_fn = tf.keras.losses.MSE,
            classifier_loss_fn =  BinaryCrossEntropy,
            metric = accuracy,
            n_mini_batches = 10,
            unsupervised_weight=10
        )

        self.task += '_SemiSupervised'
        self.output_dir = os.path.join(self.base_project_dir, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.TrainedModels_dir = os.path.join(self.base_project_dir, 'TrainedModels')
        if not os.path.exists(self.TrainedModels_dir):
            os.mkdir(self.TrainedModels_dir)
        
        self.TrainedModels_dir = os.path.join(self.TrainedModels_dir, self.task)
        if not os.path.exists(self.TrainedModels_dir):
            os.mkdir(self.TrainedModels_dir)

        model.encoder.save_weights('initial_encoder_weights.h5')
        model.decoder.save_weights('initial_decoder_weights.h5')
        model.classifier.save_weights('initial_classifier_weights.h5')
        ten_fold = utils.Dataset10FoldSpliter(self.X, self.Y, shuffle=True, seed=self.seed)
        Y_test_set = list()
        Y_pred_set = list()
        for i in range(10):
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.test_tracker = ten_fold.split()
        
            print('\n\n')
            n_X_train = self.X_train.shape[0]
            temp_X = np.vstack((self.X_train, self.unlabeled_X))
            temp_X, self.X_val, ss = utils.preprocess_signals(temp_X, self.X_val, True)
            self.X_test = utils.apply_standardizer(self.X_test, ss)

            self.X_train = temp_X[:n_X_train]
            self.unlabeled_X = temp_X[n_X_train:]

            data = {
                'labeled_signals':self.X_train,
                'unlabeled_signals':self.unlabeled_X,
                'train_labels':self.Y_train,
                'val_signals':self.X_val,
                'val_labels':self.Y_val
            }

            print('Training ' + str(i+1) + 'th model. Train 0 class: ' + str(self.X_train.shape[0] - np.sum(self.Y_train)) + ' - Test 0 class: ' + str(self.X_test.shape[0] - np.sum(self.Y_test)))
            model.encoder.load_weights('initial_encoder_weights.h5')
            model.decoder.load_weights('initial_decoder_weights.h5')
            model.classifier.load_weights('initial_classifier_weights.h5')
            try:
                model.fit(
                    data,
                    epochs=500,
                )
            except KeyboardInterrupt:
                pass
            test_encoder_output = model.encoder(self.X_test)
            Y_pred = model.classifier(test_encoder_output)

            Y_test_set.append(self.Y_test)
            Y_pred_set.append(Y_pred)

            save_results_to = os.path.join(self.output_dir, self.task + '_fold' + str(i) + '.txt')
            print_and_save_results(self.Y_test, Y_pred, save_to=save_results_to)

            encode_path = os.path.join(self.TrainedModels_dir, 'encode' + str(i) + '.h5')
            decode_path = os.path.join(self.TrainedModels_dir, 'decode' + str(i) + '.h5')
            classifier_path = os.path.join(self.TrainedModels_dir, 'classifier' + str(i) + '.h5')
            
            model.encoder.save_weights(encode_path)
            model.decoder.save_weights(decode_path)
            model.classifier.save_weights(classifier_path)
            

        print(50*"=", 'Final Results on 10-Fold', 50*"=", end='n\n')
        self.evaluate(Y_test_set, Y_pred_set)
        os.remove("initial_encoder_weights.h5")
        os.remove("initial_decoder_weights.h5")
        os.remove("initial_classifier_weights.h5")

    def evaluate(self, Y_test_set, Y_pred_set):
        Y_pred_total = Y_pred_set[0]
        Y_total = Y_test_set[0]
        for i in range(1, len(Y_pred_set)):
            Y_pred_total = np.vstack((Y_pred_total, Y_pred_set[i]))
            Y_total = np.vstack((Y_total, Y_test_set[i]))

        save_results_to = os.path.join(self.output_dir, self.task + '.txt')
        print_and_save_results(Y_total, Y_pred_total, save_to=save_results_to)