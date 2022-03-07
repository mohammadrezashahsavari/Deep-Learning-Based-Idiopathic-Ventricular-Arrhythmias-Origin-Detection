import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback): 
    def __init__(self, metric='f1', threshold=0.99):
        self.metric = metric
        self.threshold = threshold
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get(self.metric) <= self.threshold):   
            print("\nReached %2.2f%% loss. Stopping training!!" %(self.threshold*100))   
            self.model.stop_training = True

