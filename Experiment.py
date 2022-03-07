import os
from experiments.supervised import *
from experiments.semisupervised import *

base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

dataset_path = os.path.join(base_project_dir, 'Data', 'ZeroPaddedFiltered50HzDataset')


model = Experiment(dataset_path, network_structure='VGG_BiLSTM', plot_attention_weights=False, use_pre_trained=False, seed=300)#, augmentition={'type':'rescaling', 'params':[[0.8, 1.2], 5]})
model.prepare()
model.train_10fold()
#model.reproduce_results_on_10fold()
exit(0)


'''
model = SemiSupervisedExperiment(dataset_path, plot_attention_weights=False, use_pre_trained=False, seed=300)#, augmentition={'type':'shifting', 'params':[0.1, 2]})
model.prepare()
model.train_10fold()
'''


'''
model = Experiment_RUSBoost(dataset_path, network_structure='VGG_BiLSTM')
model.prepare()
model.train()
'''
