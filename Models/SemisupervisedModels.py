import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import Model

def EncoderModel(input_shape, n_output_nodes=1):
    inputs = Input(shape=input_shape)
    '''t = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    t = Conv1D(64, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Bidirectional(LSTM(64, return_sequences=True))(t)
    features, forward_state_h, forward_state_c, backward_state_h, backward_state_c = Bidirectional(LSTM(64, return_sequences=True, return_state=True))(t)
    state_h = Concatenate()([forward_state_h, backward_state_h])'''
    t = LSTM(64, return_sequences=True)(inputs)
    features, state_h, state_c = LSTM(64, return_sequences=True, return_state=True)(t)


    encoder = Model(inputs, [features, state_h])
    encoder._name  = 'encoder'

    return encoder


def DecoderModel(Encoder, output_shape):
    encoder_output_features_shape = Encoder.outputs[0].shape[1:]

    featueres_inputs = Input(shape=encoder_output_features_shape)
    t = LSTM(64, return_sequences=True)(featueres_inputs)
    t = LSTM(64, return_sequences=True)(t)
    t = Dense(output_shape[1])(t)
    '''t = Bidirectional(LSTM(64, return_sequences=True))(featueres_inputs)
    t = Bidirectional(LSTM(64, return_sequences=True))(t)
    t = Conv1DTranspose(512, 3, padding='same', activation='relu')(t)
    t = Conv1DTranspose(512, 3, padding='same', activation='relu')(t)
    t = Conv1DTranspose(512, 3, padding='same', activation='relu')(t)
    t = UpSampling1D(2)(t)
    t = Conv1DTranspose(512, 3, padding='same', activation='relu')(t)
    t = Conv1DTranspose(512, 3, padding='same', activation='relu')(t)
    t = Conv1DTranspose(512, 3, padding='same', activation='relu')(t)
    t = UpSampling1D(2)(t)
    t = Conv1DTranspose(256, 3, padding='same', activation='relu')(t)
    t = Conv1DTranspose(256, 3, padding='same', activation='relu')(t)
    t = Conv1DTranspose(256, 3, padding='same', activation='relu')(t)
    t = UpSampling1D(2)(t)
    t = Conv1DTranspose(128, 3, padding='same', activation='relu')(t)
    t = Conv1DTranspose(128, 3, padding='same', activation='relu')(t)
    t = UpSampling1D(2)(t)
    t = Conv1DTranspose(64, 3, padding='same', activation='relu')(t)
    t = Conv1DTranspose(64, 3, padding='same', activation='relu')(t)
    t = Conv1DTranspose(output_shape[1], 3, padding='same')(t)'''

    encoder_decoder = Model(featueres_inputs, t)
    encoder_decoder._name  = 'decoder'

    return encoder_decoder    


def classifierModel(Encoder, n_output_nodes):
    encoder_output_features_shape = Encoder.outputs[0].shape[1:]
    encoder_output_state_h_shape = Encoder.outputs[1].shape[1:]

    features_inputs = Input(shape=encoder_output_features_shape)
    state_h_inputs = Input(shape=encoder_output_state_h_shape)
    context_vector, attention_weights = Attention(64)(features_inputs, state_h_inputs)
    t = tf.keras.layers.Dense(256)(context_vector)
    t = tf.keras.layers.LeakyReLU(alpha=0.01)(t)
    t = tf.keras.layers.Dropout(0.2)(t)
    t = tf.keras.layers.Dense(256)(t)
    t = tf.keras.layers.LeakyReLU(alpha=0.2)(t)
    t = tf.keras.layers.Dropout(0.2)(t)
    t = Dense(n_output_nodes, activation='sigmoid' if n_output_nodes == 1 else 'softmax')(t)

    classifier = Model([features_inputs, state_h_inputs], t)
    classifier._name  = 'classifier'

    return classifier


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
          
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
          
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights





class SemiSupervisedModel():
    def __init__(
        self,
        encoder,
        decoder,
        classifier,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def compile(self, optimizer, en_de_loss_fn, classifier_loss_fn, metric, n_mini_batches, unsupervised_weight):
        self.optimizer = optimizer
        self.en_de_loss_fn = en_de_loss_fn
        self.classifier_loss_fn = classifier_loss_fn
        self.metric = metric
        self.n_mini_batches = n_mini_batches
    
    def fit(self, data, epochs):
        labeled_signals = data['labeled_signals']
        unlabeled_signals = data['unlabeled_signals']
        train_labels = data['train_labels']
        val_signals = data['val_signals']
        val_labels = data['val_labels']


        n_labeled_signals = labeled_signals.shape[0]
        n_unlabeled_signals = unlabeled_signals.shape[0]
        labeled_mini_batch_size = n_labeled_signals // self.n_mini_batches
        unlabeled_mini_batch_size = n_unlabeled_signals // self.n_mini_batches

        sup_weights = 1
        add_sup_weight_on_epoch_end = 0.2
        for epoch in range(epochs):
            for i in range(self.n_mini_batches):
                # Unsupervised Training
                # Create minibatches
                if i < self.n_mini_batches-1:
                    labeled_mini_batch_signals = labeled_signals[i*labeled_mini_batch_size:(i+1)*labeled_mini_batch_size]
                    mini_batch_labels = train_labels[i*labeled_mini_batch_size:(i+1)*labeled_mini_batch_size].reshape(-1, 1)
                    unlabeled_mini_batch_signals = unlabeled_signals[i*unlabeled_mini_batch_size:(i+1)*unlabeled_mini_batch_size]
                elif i == self.n_mini_batches-1:
                    labeled_mini_batch_signals = labeled_signals[i*labeled_mini_batch_size:]
                    mini_batch_labels = train_labels[i*labeled_mini_batch_size:].reshape(-1, 1)
                    unlabeled_mini_batch_signals = unlabeled_signals[i*unlabeled_mini_batch_size:]
                mini_batch_signals_unsup = tf.concat([labeled_mini_batch_signals, unlabeled_mini_batch_signals], axis=0) 

                with tf.GradientTape(persistent=True) as tape:
                    encoder_output_sup = self.encoder(labeled_mini_batch_signals)
                    encoder_output_unsup = self.encoder(mini_batch_signals_unsup)
                    pred_labels = self.classifier(encoder_output_sup) + 1e-8   # adding epsilon to address divergence issuses
                    reconstructed_mini_batch_signals = self.decoder(encoder_output_unsup[0])
                    en_de_loss = self.en_de_loss_fn(mini_batch_signals_unsup, reconstructed_mini_batch_signals)
                    classifier_loss = self.classifier_loss_fn(mini_batch_labels, pred_labels, class_weights={'0':3, '1':1})
                    loss = sup_weights*classifier_loss + tf.reduce_sum(en_de_loss)
                # Get the gradients of 3 models  loss
                gradients = tape.gradient(loss, [self.encoder.trainable_variables, self.decoder.trainable_variables, self.classifier.trainable_variables])
     
                self.optimizer.apply_gradients(
                    zip(gradients[0], self.encoder.trainable_variables)
                )
                self.optimizer.apply_gradients(
                    zip(gradients[1], self.decoder.trainable_variables)
                )
                self.optimizer.apply_gradients(
                    zip(gradients[2], self.classifier.trainable_variables)
                )
            sup_weights += add_sup_weight_on_epoch_end

            val_encoder_output = self.encoder(val_signals)
            val_pred_labeld = self.classifier(val_encoder_output) + 1e-8
            val_loss = self.classifier_loss_fn(val_labels, val_pred_labeld)
            train_encoder_output = self.encoder(labeled_signals)
            train_pred_labels = self.classifier(train_encoder_output)
            train_metric = self.metric(train_labels, train_pred_labels)
            val_metric = self.metric(val_labels, val_pred_labeld)
            print('Epoch:' + str(epoch+1) + '/' + str(epochs))
            print(" ========> en_de_loss:" + str(tf.reduce_sum(en_de_loss).numpy()), " - classifier_loss:" + str(classifier_loss.numpy()), ' - accuracy:' + str(train_metric.numpy()), ' - val_loss:' + str(val_loss.numpy()), ' - val_accuracy:' + str(val_metric.numpy()))



