import os
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def normalize_signal(tSignal):
      # copy the data if needed, omit and rename function argument if desired
      signal = np.copy(tSignal) # signal is in range [a;b]
      minimum = np.min(signal)
      maximum = np.max(signal)
      signal -= minimum # signal is in range to [0;b-a]
      signal /= maximum-minimum # signal is normalized to [0;1]
      return signal

def create_mat_dataset(dataset_dir, mat_dataset_dir, sampleing_rate=2000, normalize=False):
  if not os.path.exists(mat_dataset_dir):
    os.mkdir(mat_dataset_dir)

  signal_names = os.listdir(dataset_dir)

  for signal_name in tqdm(signal_names):
    ecg_signal_path = os.path.join(dataset_dir, signal_name)
    ecg_signal = pd.read_csv(ecg_signal_path)
    ecg_signal = ecg_signal.to_numpy()
    if normalize:
      for channel in range(12):
        ecg_signal[:, channel] = normalize_signal(ecg_signal[:, channel])

    '''time_axies_org = list(map(lambda x:x/sampleing_rate, [i for i in range(ecg_signal.shape[0])]))
    plt.xlabel('Time(s)')
    plt.ylabel('Padded ECG signal')
    plt.plot(time_axies_org, ecg_signal[:, 0])
    plt.show()
    continue'''
 
    mat_file_path = os.path.join(mat_dataset_dir, signal_name.split('.')[0] + '.mat')
    mat_file_content = {'ecg_signal':ecg_signal}
    savemat(mat_file_path, mat_file_content)


if __name__ == '__main__':
  base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  dataset_dir = os.path.join(base_project_dir, 'Data', 'PVCVTECGData')
  mat_dataset_dir = os.path.join(base_project_dir, 'Data', 'MatPVCVTECGData')

  sampleing_rate = 2000

  create_mat_dataset(dataset_dir, mat_dataset_dir, sampleing_rate, False)