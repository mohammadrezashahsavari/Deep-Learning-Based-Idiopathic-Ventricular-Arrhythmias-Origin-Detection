import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa.core import resample
from random import uniform

base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(base_project_dir)



def random_rescale_signal(ecg_signal, from_sample, to_samples):
    #org_n_samples = ecg_signal.shape[0]
    #rescaled_n_samples = int(rate*org_n_samples)
    return resample(ecg_signal, from_sample, to_samples)[:to_samples]


def random_rescale_dataset(mat_dataset_dir, n_out_of_each=50, rescale_range=[0.8, 1.2], sampling_rate=2000):
    rescaled_dataset_dir = os.path.join(base_project_dir, 'Data', 'RescaledDataset')
    database_path = os.path.join(base_project_dir, 'Data', 'Diagnosis.csv')
    database = pd.read_csv(database_path, index_col='HospitalID')
    if not os.path.exists(rescaled_dataset_dir):
        os.mkdir(rescaled_dataset_dir)

    augmeted_database_list = list()
    for index, row in tqdm(database.iterrows()):
        ecg_signal_path = os.path.join(mat_dataset_dir, str(index) + '.mat')
        ecg_signal = loadmat(ecg_signal_path).get('ecg_signal')
        augmeted_database_list.append([
            str(index) + '_0',
            row['Type'],
            row['LeftRight'],
            row['Sublocation'],
            row['Gender'],
            1,
        ])

        mat_file_path = os.path.join(rescaled_dataset_dir, str(index) + '_0.mat')
        mat_file_content = {'ecg_signal':ecg_signal}
        savemat(mat_file_path, mat_file_content)
        for i_th_aug in range(n_out_of_each):
            rate = uniform(*rescale_range)
            org_n_samples = ecg_signal.shape[0]
            rescaled_n_samples = int(rate*org_n_samples)
            rescaled_ecg_signal = np.zeros((rescaled_n_samples, 12))
            for c in range(12):
                rescaled_ecg_signal[:, c] = random_rescale_signal(ecg_signal[:, c], org_n_samples, rescaled_n_samples)

            '''rescaled_n_sampled = rescaled_ecg_signal.shape[0]
            time_axies_org = list(map(lambda x:x/sampling_rate, [i for i in range(org_n_samples)]))
            time_axies_rescaled = list(map(lambda x:x/sampling_rate, [i for i in range(rescaled_n_sampled)]))
            plt.xlabel('Time(s)')
            plt.ylabel('Padded ECG signal')
            plt.plot(time_axies_org, ecg_signal[:, 0], time_axies_rescaled, rescaled_ecg_signal[:, 0])
            plt.legend(['Org signal', 'Rescaled signal with:%.2f factor'%rate])
            plt.show()
            #input()
            continue'''
            augmeted_database_list.append([
                    str(index) + '_' + str(i_th_aug + 1),
                    row['Type'],
                    row['LeftRight'],
                    row['Sublocation'],
                    row['Gender'],
                    rate,
                ])

            mat_file_path = os.path.join(rescaled_dataset_dir, str(index) + '_' + str(i_th_aug + 1) + '.mat')
            mat_file_content = {'ecg_signal':rescaled_ecg_signal}
            savemat(mat_file_path, mat_file_content)

    columns = ['HospitalID', 'Type', 'LeftRight', 'Sublocation', 'Gender', 'Rate']
    augmented_database = pd.DataFrame(augmeted_database_list, columns=columns)
    augmanted_database_path = os.path.join(mat_dataset_dir, '..', 'augmented_database.csv')
    augmented_database.to_csv(augmanted_database_path)



if __name__ =='__main__':
    
    mat_dataset_dir = os.path.join(base_project_dir, 'Data', 'MatPVCVTECGData')
    random_rescale_dataset(mat_dataset_dir, 50, [0.8, 1.2], 2000)


