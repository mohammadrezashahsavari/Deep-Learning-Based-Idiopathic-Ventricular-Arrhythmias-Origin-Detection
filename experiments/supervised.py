import os
from types import new_class
import utils
import numpy as np
import pandas as pd
from Models import models, SemisupervisedModels
from tools.metrics import *
from tools.losses import *
from tools import callbacks
from random import shuffle
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from librosa.core import resample
import json

class Experiment():
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
        tf.keras.utils.set_random_seed(seed)

    def prepare(self):
        if not self.database_path:
            self.database_path = os.path.join(self.dataset_path, '..', 'Diagnosis.csv')

        self.database = utils.load_database(self.database_path)
        self.database = self.database[:]
        self.Y, database, self.lable_encoder = utils.preprocess_lables(self.database, self.diag_mode)
        self.X = utils.load_raw_signals(self.dataset_path, database)
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
            
        print('Input Shape:', self.input_shape)
        print('Number of Classes:', self.n_classes)
        

    def train(self):
        n_output_nodes = 1 if self.n_classes == 2 else self.n_classes
        if self.network_structure == 'Shallow_CNN_BiLSTM':
            self.classifier = models.Shallow_CNN_BiLSTM_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'discriminator':
            self.classifier = models.discriminator_Moldel(self.input_shape)
        elif self.network_structure == 'VGG':
            self.classifier = models.VGG_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_LSTM':
            self.classifier = models.VGG_LSTM_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_BiLSTM':
            self.classifier = models.VGG_BiLSTM_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_LSTM_Attn':
            self.classifier = models.VGG_LSTM_Attn_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_BiLSTM_Attn':
            self.classifier = models.VGG_BiLSTM_Attn_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'Resnet18_LSTM_Attn':
            self.classifier = models.Resnet18_LSTM_Attn_Model(input_shape=self.input_shape, n_output_nodes=n_output_nodes)
        else:
            raise ValueError('Network Structure not supported.\
                try one of these structures: 1-VGG  2-VGG_LSTM  3-VGG_LSTM_Attn, 4-VGG_BiLSTM_Attn, 5-Resnet18_LSTM_Attn\
                    ')
        self.task = self.task + '_' + self.network_structure

        if self.use_pre_trained:
            org_pretrained_model_path = os.path.join(self.base_project_dir, 'PreTrainedModels', self.network_structure + '_PreTrained' + '.h5')
            print("Pretrained model loaded successfuly.")
            self.classifier.load_weights(org_pretrained_model_path)

            for layer in self.classifier.layers:
                layer.trainable = True

            base_model_output = self.classifier.get_layer('bidirectional_1').output
            t = tf.keras.layers.Dense(256)(base_model_output)
            t = tf.keras.layers.LeakyReLU(alpha=0.01)(t)
            t = tf.keras.layers.Dropout(0.2)(t)
            t = tf.keras.layers.Dense(256)(t)
            t = tf.keras.layers.LeakyReLU(alpha=0.01)(t)
            t = tf.keras.layers.Dropout(0.2)(t)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(t)
            self.classifier = tf.keras.Model(self.classifier.inputs, outputs)
            for layer in self.classifier.layers[:7]:
                layer.trainable = False
            self.task += '_PreTrained'

            '''for layer in self.classifier.layers[:-3]:
                layer.trainable = False
            '''

        callback1 = tf.keras.callbacks.EarlyStopping(patience=50)
        callback2 = callbacks.MyCallback('val_loss', 0.0)

        opt = tf.keras.optimizers.Adam(learning_rate=0.00005, clipvalue=5.0)

        self.classifier.summary()
        
        self.classifier.compile(
            optimizer=opt,
            loss='binary_crossentropy' if self.n_classes == 2 else 'categorical_crossentropy',
            metrics=f1 if self.n_classes == 2 else f1_multi_class
        )

        self.output_dir = os.path.join(self.base_project_dir, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        output_models_dir = os.path.join(self.base_project_dir, 'output_models')
        if not os.path.exists(output_models_dir):
            os.mkdir(output_models_dir)

        output_models_dir = os.path.join(output_models_dir, self.task)
        if not os.path.exists(output_models_dir):
            os.mkdir(output_models_dir)

        print('Task:', self.task)

        self.X_train_val, self.X_test, self.Y_train_val, self.Y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=self.seed)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train_val, self.Y_train_val, test_size=0.12, random_state=sedl.seed)

        if self.augmentition_type == 'shifting':
            self.X_train, self.Y_train = utils.augment_signals_by_shifting(self.X_train, self.Y_train, *self.augmentition_params, self.sampleing_rate, self.seed)
            self.X_val = utils.crop_test_signals(self.X_val, *self.augmentition_params, self.input_shape, self.sampleing_rate)
            self.X_test = utils.crop_test_signals(self.X_test, *self.augmentition_params, self.input_shape, self.sampleing_rate)
        elif self.augmentition_type == 'rescaling':
            self.X_train, self.Y_train = utils.augment_signals_by_rescaling(self.X_train, self.Y_train, self.augmentition_params[0], self.augmentition_params[1], self.sampleing_rate, self.seed)
            self.X_val = utils.self_padd_test_signals(self.X_val, self.augmentition_params[0])
            self.X_test = utils.self_padd_test_signals(self.X_test, self.augmentition_params[0])

        print('\n\n')    
        self.X_train, self.X_val, ss = utils.preprocess_signals(self.X_train, self.X_val, True)
        self.X_test = utils.apply_standardizer(self.X_test, ss)

        history_dir = os.path.join(self.base_project_dir, 'Histories')
        if not os.path.exists(history_dir):
            os.mkdir(history_dir)
        history_path = os.path.join(history_dir, self.task + '.json')
        try :
            history = self.classifier.fit(
                self.X_train,
                self.Y_train,
                epochs=1000,
                batch_size=32,
                callbacks=[callback1, callback2],
                validation_data=(self.X_val, self.Y_val),
            )
            with open(history_path, 'w') as file_pi:
                json.dump(history.history, file_pi)
        except KeyboardInterrupt:
            pass

        if self.plot_attention_weights:
            use_lead = 0
            sampling_rate = 50
            attention_weight_predictor = tf.keras.Model(self.classifier.inputs, self.classifier.get_layer('attention').output[1])  
            attention_weight_test = attention_weight_predictor(self.X_test).numpy()
            for test_ecg_idx in range(self.X_test.shape[0]):
                attention_weight = resample(attention_weight_test[test_ecg_idx, :, use_lead], attention_weight_test.shape[1], self.X_test.shape[1])[:self.X_test.shape[1]]
                attention_weight = attention_weight * 100
                time_axies = list(map(lambda x:x/sampling_rate, [i for i in range(self.X_test.shape[1])]))
                plt.figure()
                plt.plot(time_axies, self.X_test[test_ecg_idx, :, use_lead])
                plt.plot(time_axies, attention_weight)
                plt.title('ECG Signal with Attention Weights')
                plt.legend(['ECG Signal', 'Attention Weights'])
                plt.show()
                if input('Enter \'q\' for pass the attention weight plots.') == 'q':
                    break
        

        save_results_to = os.path.join(self.output_dir, self.task + '.txt')
        print_and_save_results(self.Y_test, self.classifier(self.X_test), save_to=save_results_to)

        output_moedel_path = os.path.join(output_models_dir, self.task + '.h5')
        self.classifier.save_weights(output_moedel_path)

    
    def train_10fold(self):
        n_output_nodes = 1 if self.n_classes == 2 else self.n_classes
        if self.network_structure == 'Shallow_CNN_BiLSTM_Attn':
            self.classifier = models.Shallow_CNN_BiLSTM_Attn_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'discriminator':
            self.classifier = models.discriminator_Moldel(self.input_shape)
        elif self.network_structure == 'VGG':
            self.classifier = models.VGG_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_LSTM':
            self.classifier = models.VGG_LSTM_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_BiLSTM':
            self.classifier = models.VGG_BiLSTM_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_LSTM_Attn':
            self.classifier = models.VGG_LSTM_Attn_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_BiLSTM_Attn':
            self.classifier = models.VGG_BiLSTM_Attn_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'Resnet_BLSTM_Attn':
            self.classifier = models.Resnet_BiLSTM_Attn_Model(input_shape=self.input_shape, n_output_nodes=n_output_nodes)
        else:
            raise ValueError('Network Structure not supported.\
                try one of these structures: 1-VGG  2-VGG_LSTM  3-VGG_LSTM_Attn, 4-VGG_BiLSTM_Attn, 5-Resnet18_LSTM_Attn\
                    ')
        self.task = self.task + '_' + self.network_structure

        if self.use_pre_trained:
            org_pretrained_model_path = os.path.join(self.base_project_dir, 'PreTrainedModels', self.network_structure + '_PreTrained' + '.h5')
            print("Pretrained model loaded successfuly.")
            self.classifier.load_weights(org_pretrained_model_path)

            # for discriminator
            for layer in self.classifier.layers:
                layer.trainable = True

            base_model_output = self.classifier.get_layer('attention').output[0]
            t = tf.keras.layers.Dense(256, activation='relu')(base_model_output)
            t = tf.keras.layers.LeakyReLU(alpha=0.01)(t)
            t = tf.keras.layers.Dropout(0.2)(t)
            t = tf.keras.layers.Dense(256, activation='relu')(t)
            t = tf.keras.layers.LeakyReLU(alpha=0.01)(t)
            t = tf.keras.layers.Dropout(0.2)(t)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(t)
            self.classifier = tf.keras.Model(self.classifier.inputs, outputs)
            self.task += '_PreTrained'

            '''for layer in self.classifier.layers[:-3]:
                layer.trainable = False
            '''

        callback1 = tf.keras.callbacks.EarlyStopping(patience=50)
        callback2 = callbacks.MyCallback('loss', 0.05)

        opt = tf.keras.optimizers.Adam(learning_rate=0.00005, clipvalue=5.0)

        self.classifier.summary()
        
        self.classifier.compile(
            optimizer=opt,
            loss=BinaryCrossEntropy,    #'binary_crossentropy' if self.n_classes == 2 else 'categorical_crossentropy',
            metrics='accuracy'    #f1 if self.n_classes == 2 else f1_multi_class
        )

        self.output_dir = os.path.join(self.base_project_dir, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.output_dir = os.path.join(self.output_dir, self.task)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        TrainedModels_dir = os.path.join(self.base_project_dir, 'TrainedModels')
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        TrainedModels_dir = os.path.join(TrainedModels_dir, self.task)
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        print('Task:', self.task)

        self.classifier.save_weights('initial_weight.h5')
        # Smave Y_true
        Y_true = pd.DataFrame()
        Y_true = pd.concat([Y_true, pd.DataFrame(self.Y)], axis=0)
        Y_true.to_csv('Y_true.csv')

        Y_test_set = list()
        Y_pred_set = list()
        names_and_Y_preds = pd.DataFrame()
        ten_fold = utils.Dataset10FoldSpliter(self.X, self.Y, shuffle=True, seed=self.seed)

        epoches_list = [85, 75, 100, 120, 125, 110, 100, 85, 110, 110]
        for i in range(10):
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.test_tracker = ten_fold.split()
            if self.augmentition_type == 'shifting':
                self.X_train, self.Y_train = utils.augment_signals_by_shifting(self.X_train, self.Y_train, *self.augmentition_params, self.sampleing_rate, self.seed)
                self.X_val = utils.crop_test_signals(self.X_val, *self.augmentition_params, self.input_shape, self.sampleing_rate)
                self.X_test = utils.crop_test_signals(self.X_test, *self.augmentition_params, self.input_shape, self.sampleing_rate)
            elif self.augmentition_type == 'rescaling':
                self.X_train, self.Y_train = utils.augment_signals_by_rescaling(self.X_train, self.Y_train, self.augmentition_params[0], self.augmentition_params[1], self.sampleing_rate, self.seed)
                self.X_val = utils.self_padd_test_signals(self.X_val, self.augmentition_params[0])
                self.X_test = utils.self_padd_test_signals(self.X_test, self.augmentition_params[0])

            print('\n\n')    
            self.X_train, self.X_val, ss = utils.preprocess_signals(self.X_train, self.X_val, True)
            self.X_test = utils.apply_standardizer(self.X_test, ss)

            print('Training ' + str(i+1) + 'th model. Train 0 class: ' + str(self.X_train.shape[0] - np.sum(self.Y_train)) + ' - Test 0 class: ' + str(self.X_test.shape[0] - np.sum(self.Y_test)))
            tf.keras.backend.clear_session()
            self.classifier.load_weights("initial_weight.h5")
            
            try :
                self.classifier.fit(
                    self.X_train,
                    self.Y_train,
                    epochs=epoches_list[i],
                    batch_size=32,
                    #callbacks=[callback2],
                    validation_data=(self.X_val, self.Y_val),
                )
            except KeyboardInterrupt:
                pass
            
            Y_pred = self.classifier(self.X_test)

            Y_test_set.append(self.Y_test)
            Y_pred_set.append(Y_pred)

            # Save names and Y_pres to a dataframe
            names_and_Y_preds_fold = np.hstack((self.test_tracker, Y_pred))
            names_and_Y_preds = pd.concat([names_and_Y_preds, pd.DataFrame(names_and_Y_preds_fold)], axis=0)

            correct_preds = (K.round(Y_pred) == self.Y_test)
            names_and_correct_preds = np.hstack((self.test_tracker, correct_preds))
            print(names_and_correct_preds)

            save_results_to = os.path.join(self.output_dir, self.task + '_fold' + str(i+1) + '.txt')
            print_and_save_results(self.Y_test, Y_pred, save_to=save_results_to)

            output_moedel_path = os.path.join(TrainedModels_dir, self.task + str(i) + '.h5')
            self.classifier.save_weights(output_moedel_path)

            if self.plot_attention_weights:
                use_lead = 0
                sampling_rate = 50
                attention_weight_predictor = tf.keras.Model(self.classifier.inputs, self.classifier.get_layer('attention').output[1])  
                attention_weight_test = attention_weight_predictor(self.X_test).numpy()
                for test_ecg_idx in range(self.X_test.shape[0]):
                    attention_weight = resample(attention_weight_test[test_ecg_idx, :, use_lead], attention_weight_test.shape[1], self.X_test.shape[1])[:self.X_test.shape[1]]
                    attention_weight = attention_weight * 100
                    time_axies = list(map(lambda x:x/sampling_rate, [i for i in range(self.X_test.shape[1])]))
                    plt.figure()
                    plt.plot(time_axies, self.X_test[test_ecg_idx, :, use_lead], linewidth=2, color='black')
                    plt.fill_between(time_axies, attention_weight, step="pre", color='red', alpha=0.2)
                    plt.title('ECG Signal with Attention Weights')
                    plt.legend(['ECG Signal', 'Attention Weights'])
                    plt.show()
                    if input('Enter \'q\' for pass the attention weight plots.') == 'q':
                        break

        names_and_Y_preds.columns = ['Names', 'Y_preds']
        names_and_Y_preds = names_and_Y_preds.sort_values(by=['Names'])
        names_and_Y_preds_csv_path = os.path.join(self.output_dir, self.task + '_Names&Y_preds' + '.csv')
        names_and_Y_preds.to_csv(names_and_Y_preds_csv_path)
        print(50*"=", 'Final Results on 10-Fold', 50*"=")
        self.evaluate(Y_test_set, Y_pred_set)
        #os.remove("initial_weight.h5")
    

    def reproduce_results_on_10fold(self):
        n_output_nodes = 1 if self.n_classes == 2 else self.n_classes
        if self.network_structure == 'Shallow_CNN_BiLSTM_Attn':
            self.classifier = models.Shallow_CNN_BiLSTM_Attn_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'discriminator':
            self.classifier = models.discriminator_Moldel(self.input_shape)
        elif self.network_structure == 'VGG':
            self.classifier = models.VGG_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_LSTM':
            self.classifier = models.VGG_LSTM_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_BiLSTM':
            self.classifier = models.VGG_BiLSTM_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_LSTM_Attn':
            self.classifier = models.VGG_LSTM_Attn_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_BiLSTM_Attn':
            self.classifier = models.VGG_BiLSTM_Attn_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'Resnet_BLSTM_Attn':
            self.classifier = models.Resnet_BiLSTM_Attn_Model(input_shape=self.input_shape, n_output_nodes=n_output_nodes)
        else:
            raise ValueError('Network Structure not supported.\
                try one of these structures: 1-VGG  2-VGG_LSTM  3-VGG_LSTM_Attn, 4-VGG_BiLSTM_Attn, 5-Resnet18_LSTM_Attn\
                    ')
        self.task = self.task + '_' + self.network_structure

        if self.use_pre_trained:
            org_pretrained_model_path = os.path.join(self.base_project_dir, 'PreTrainedModels', self.network_structure + '_PreTrained' + '.h5')
            print("Pretrained model loaded successfuly.")
            self.classifier.load_weights(org_pretrained_model_path)

            # for discriminator
            for layer in self.classifier.layers:
                layer.trainable = True

            #base_model = tf.keras.Model(self.classifier.inputs, self.classifier.get_layer('bidirectional_1').output)  
            base_model_output = self.classifier.get_layer('attention').output[0]
            t = tf.keras.layers.Dense(256, activation='relu')(base_model_output)
            t = tf.keras.layers.LeakyReLU(alpha=0.2)(t)
            t = tf.keras.layers.Dropout(0.2)(t)
            t = tf.keras.layers.Dense(256, activation='relu')(t)
            t = tf.keras.layers.LeakyReLU(alpha=0.2)(t)
            t = tf.keras.layers.Dropout(0.2)(t)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(t)
            self.classifier = tf.keras.Model(self.classifier.inputs, outputs)
            self.task += '_PreTrained'

        self.output_dir = os.path.join(self.base_project_dir, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        self.output_dir = os.path.join(self.output_dir, self.task)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        TrainedModels_dir = os.path.join(self.base_project_dir, 'TrainedModels')
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        TrainedModels_dir = os.path.join(TrainedModels_dir, self.task)
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        print('Task:', self.task)

        Y_test_set = list()
        Y_pred_set = list()
        names_and_Y_preds = pd.DataFrame()
        ten_fold = utils.Dataset10FoldSpliter(self.X, self.Y, shuffle=True, seed=self.seed)
        for i in range(10):
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.test_tracker = ten_fold.split()
            if self.augmentition_type == 'shifting':
                self.X_train, self.Y_train = utils.augment_signals_by_shifting(self.X_train, self.Y_train, *self.augmentition_params, self.sampleing_rate, self.seed)
                self.X_val = utils.crop_test_signals(self.X_val, *self.augmentition_params, self.input_shape, self.sampleing_rate)
                self.X_test = utils.crop_test_signals(self.X_test, *self.augmentition_params, self.input_shape, self.sampleing_rate)
            elif self.augmentition_type == 'rescaling':
                self.X_train, self.Y_train = utils.augment_signals_by_rescaling(self.X_train, self.Y_train, self.augmentition_params[0], self.augmentition_params[1], self.sampleing_rate, self.seed)
                self.X_val = utils.self_padd_test_signals(self.X_val, self.augmentition_params[0])
                self.X_test = utils.self_padd_test_signals(self.X_test, self.augmentition_params[0])

            print('\n\n')    
            self.X_train, self.X_val, ss = utils.preprocess_signals(self.X_train, self.X_val, True)
            self.X_test = utils.apply_standardizer(self.X_test, ss)

            print('Clculating ' + str(i+1) + 'th models Y_pred .')
            tf.keras.backend.clear_session()
            trained_moedel_path = os.path.join(TrainedModels_dir, self.task + str(i) + '.h5')
            self.classifier.load_weights(trained_moedel_path)
            
            Y_pred = self.classifier(self.X_test)

            Y_test_set.append(self.Y_test)
            Y_pred_set.append(Y_pred)

            # Save names and Y_pres to a dataframe
            names_and_Y_preds_fold = np.hstack((self.test_tracker, Y_pred))
            names_and_Y_preds = pd.concat([names_and_Y_preds, pd.DataFrame(names_and_Y_preds_fold)], axis=0)

            correct_preds = (K.round(Y_pred) == self.Y_test)
            names_and_correct_preds = np.hstack((self.test_tracker, correct_preds))
            print(names_and_correct_preds)

            save_results_to = os.path.join(self.output_dir, self.task + '_fold' + str(i+1) + '.txt')
            print_and_save_results(self.Y_test, Y_pred, save_to=save_results_to)

            if self.plot_attention_weights:
                font = {'family' : 'normal', 'size'   : 22}
                matplotlib.rc('font', **font)
                use_lead = 0
                sampling_rate = 50
                attention_weight_predictor = tf.keras.Model(self.classifier.inputs, self.classifier.get_layer('attention').output[1])  
                attention_weight_test = attention_weight_predictor(self.X_test).numpy()
                self.X_test = utils.reverse_standardizer(self.X_test, ss)
                for test_ecg_idx in range(self.X_test.shape[0]):
                    attention_weight = resample(attention_weight_test[test_ecg_idx, :, use_lead], attention_weight_test.shape[1], self.X_test.shape[1])[:self.X_test.shape[1]]
                    attention_weight = attention_weight * np.max(self.X_test[test_ecg_idx, :, use_lead]) * 20
                    time_axies = list(map(lambda x:x/sampling_rate, [i for i in range(self.X_test.shape[1])]))
                    plt.figure()
                    plt.plot(time_axies, self.X_test[test_ecg_idx, :, use_lead], linewidth=2, color='black')
                    plt.fill_between(time_axies, attention_weight, step="pre", color='red', alpha=0.2)
                    plt.title('ECG Signal with Attention Weights')
                    plt.legend(['ECG Signal', 'Attention Weights'])
                    plt.ylabel('µV')
                    #plt.grid(linestyle='--', linewidth=0.5)
                    plt.show()
                    '''if input('Enter \'q\' for pass the attention weight plots.') == 'q':
                        break'''

        names_and_Y_preds.columns = ['Names', 'Y_preds']
        names_and_Y_preds = names_and_Y_preds.sort_values(by=['Names'])
        names_and_Y_preds_csv_path = os.path.join(self.output_dir, self.task + '_Names&Y_preds' + '.csv')
        names_and_Y_preds.to_csv(names_and_Y_preds_csv_path)
        print(50*"=", 'Final Results on 10-Fold', 50*"=")
        self.evaluate(Y_test_set, Y_pred_set)
        #os.remove("initial_weight.h5")

    def evaluate(self, Y_test_set, Y_pred_set):
        Y_pred_total = Y_pred_set[0]
        Y_total = Y_test_set[0]
        for i in range(1, len(Y_pred_set)):
            Y_pred_total = np.vstack((Y_pred_total, Y_pred_set[i]))
            Y_total = np.vstack((Y_total, Y_test_set[i]))

        save_results_to = os.path.join(self.output_dir, self.task + '.txt')
        print_and_save_results(Y_total, Y_pred_total, save_to=save_results_to)




class Experiment_RUSBoost():
    def __init__(self, dataset_path, database_path=None, sampleing_rate=50, diag_mode='LeftRight', n_folds=10, base_project_dir='.', augmentition=None, network_structure='VGG_LSTM_Attn', start_fold=1):
        self.dataset_path = dataset_path
        self.database_path = database_path
        self.sampleing_rate = sampleing_rate
        self.diag_mode = diag_mode
        self.n_folds = n_folds
        self.base_project_dir = base_project_dir
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

        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def prepare(self):
        if not self.database_path:
            self.database_path = os.path.join(self.dataset_path, '..', 'Diagnosis.csv')

        self.database = utils.load_database(self.database_path)
        self.database = self.database[:]
        self.Y, database, self.lable_encoder = utils.preprocess_lables(self.database, self.diag_mode)
        self.X = utils.load_raw_signals(self.dataset_path, database)

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
            
        print('Input Shape:', self.input_shape)
        print('Number of Classes:', self.n_classes)
        
    def train(self):
        n_output_nodes = 1 if self.n_classes == 2 else self.n_classes
        if self.network_structure == 'VGG':
            self.classifier = models.VGG_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_LSTM':
            self.classifier = models.VGG_LSTM_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_BiLSTM':
            self.classifier = models.VGG_BiLSTM_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_LSTM_Attn':
            self.classifier = models.VGG_LSTM_Attn_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'VGG_BiLSTM_Attn':
            self.classifier = models.VGG_BiLSTM_Attn_Model(self.input_shape, n_output_nodes)
        elif self.network_structure == 'Resnet18_LSTM_Attn':
            self.classifier = models.Resnet18_LSTM_Attn_Model(input_shape=self.input_shape, n_output_nodes=n_output_nodes)
        else:
            raise ValueError('Network Structure not supported.\
                try one of these structures: 1-VGG  2-VGG_LSTM  3-VGG_LSTM_Attn, 4-VGG_BiLSTM_Attn, 5-Resnet18_LSTM_Attn\
                    ')
        self.task = self.task + '_' + self.network_structure

        callback = tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=5.0)

        self.classifier.summary()
        
        self.classifier.compile(
            optimizer=opt,
            loss='binary_crossentropy' if self.n_classes == 2 else 'categorical_crossentropy',
            metrics=f1 if self.n_classes == 2 else f1_multi_class
        )
        self.output_dir = os.path.join(self.base_project_dir, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        print('Task:', self.task)

        rus = RUSBoost(X=self.X, Y=self.Y, n_folds=10, n_classifiers=5, weak_learner=self.classifier, min_ration=0.5, training_parameters_dic={'epochs':1000, 'batch_size':64, 'callbacks':[callback,]}, base_project_dir=self.base_project_dir, output_dir=self.output_dir, task=self.task)
        rus.perform()
        
    def evaluate(self, Y_test_set, Y_pred_set):
        Y_pred_total = Y_pred_set[0]
        Y_total = Y_test_set[0]
        for i in range(1, len(Y_pred_set)):
            Y_pred_total = np.vstack((Y_pred_total, Y_pred_set[i]))
            Y_total = np.vstack((Y_total, Y_test_set[i]))

        save_results_to = os.path.join(self.output_dir, self.task + '.txt')
        print_and_save_results(Y_total, Y_pred_total, save_to=save_results_to)










