import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks

base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
sys.path.append(base_project_dir)




def zero_padd_signal(ecg_signal, padd_up_to=25, sampling_rate=2000):
    org_n_samples = ecg_signal.shape[0]
    padded_n_sampled = sampling_rate * padd_up_to
    padded_ecg_signal = np.zeros((padded_n_sampled, 12))
    padded_ecg_signal[:org_n_samples, :] = ecg_signal
    return padded_ecg_signal

def zero_padd_dataset(mat_dataset_dir, padd_up_to=25, sampling_rate=2000):
    padded_dataset_dir = os.path.join(base_project_dir, 'Data', 'ZeroPaddedMatDataset')
    if not os.path.exists(padded_dataset_dir):
        os.mkdir(padded_dataset_dir)

    signal_names = os.listdir(mat_dataset_dir)
    for signal_name in tqdm(signal_names):
        ecg_signal_path = os.path.join(mat_dataset_dir, signal_name)

        ecg_signal = loadmat(ecg_signal_path).get('ecg_signal')
        padded_ecg_signal = zero_padd_signal(ecg_signal, padd_up_to, sampling_rate)

        org_n_samples = ecg_signal.shape[0]
        padded_n_sampled = sampling_rate * padd_up_to
        time_axies_org = list(map(lambda x:x/sampling_rate, [i for i in range(org_n_samples)]))
        time_axies_pad = list(map(lambda x:x/sampling_rate, [i for i in range(padded_n_sampled)]))
        
        #plt.xlabel('Time(s)')
        #plt.ylabel('Padded ECG signal')
        #plt.plot(time_axies_org, ecg_signal[:, 0], time_axies_pad, padded_ecg_signal[:, 0])
        #plt.show()
        #input()
        #continue

        mat_file_path = os.path.join(padded_dataset_dir, signal_name)
        mat_file_content = {'padded_ecg_signal':padded_ecg_signal}
        savemat(mat_file_path, mat_file_content)





def self_padd_signal(ecg_signal, padd_up_to=25, sampling_rate=2000):
    org_n_samples = ecg_signal.shape[0]
    padded_n_sampled = sampling_rate * padd_up_to
    padded_ecg_signal = np.zeros((padded_n_sampled, 12))
    for i in range(int(np.floor([padded_n_sampled/org_n_samples]))):
        padded_ecg_signal[org_n_samples*i: org_n_samples*(i+1), :] = ecg_signal

    start_idx = int(np.floor([padded_n_sampled/org_n_samples])) * org_n_samples
    padded_ecg_signal[start_idx:] = ecg_signal[:padded_n_sampled - start_idx, :]
    return padded_ecg_signal


def self_padd_dataset(mat_dataset_dir, padd_up_to=25, sampling_rate=2000):
    padded_dataset_dir = os.path.join(base_project_dir, 'Data', 'SlefPaddedMatDataset')
    if not os.path.exists(padded_dataset_dir):
        os.mkdir(padded_dataset_dir)

    signal_names = os.listdir(mat_dataset_dir)
    for signal_name in tqdm(signal_names):
        ecg_signal_path = os.path.join(mat_dataset_dir, signal_name)

        ecg_signal = loadmat(ecg_signal_path).get('ecg_signal')
        padded_ecg_signal = self_padd_signal(ecg_signal, padd_up_to, sampling_rate)

        '''org_n_samples = ecg_signal.shape[0]
        padded_n_sampled = sampling_rate * padd_up_to
        time_axies_org = list(map(lambda x:x/sampling_rate, [i for i in range(org_n_samples)]))
        time_axies_pad = list(map(lambda x:x/sampling_rate, [i for i in range(padded_n_sampled)]))
        plt.xlabel('Time(s)')
        plt.ylabel('Padded ECG signal')
        plt.plot(time_axies_org, ecg_signal[:, 0], time_axies_pad, padded_ecg_signal[:, 0])
        plt.show()
        #input()
        continue'''

        mat_file_path = os.path.join(padded_dataset_dir, signal_name)
        mat_file_content = {'padded_ecg_signal':padded_ecg_signal}
        savemat(mat_file_path, mat_file_content)







#=============================================================================================================================================================

def r_peak_detection(data):
    window_size = 500
    data_extended = np.concatenate([np.zeros(window_size),data,np.zeros(window_size)])
    max_list = []
    max_list_new = []  

    for i,value in enumerate(data_extended):
        if (i >= window_size) and (i < len(data_extended)-window_size):
            try:
                max_left = data_extended[(i-window_size):i+1].max()
                max_right = data_extended[i:(i+window_size)+1].max()
                check_value = data_extended[i] - ((max_left+max_right)/2)
            except ValueError:
                pass
                
            if (check_value >=0):
                max_list.append(i-window_size)
    flag = 0           
    while(flag < 2):
        flag += 1
        for i in range(1, len(max_list)-1):
            if ((data[max_list[i]]) < (data[max_list[i+1]]*9/10)) and ((data[max_list[i]])<(data[max_list[i-1]]*9/10)):
                max_list_new = np.delete(max_list, i)
                flag = 0
        if flag == 0:
            max_list = max_list_new

    return np.array(max_list)
    '''
    max_list_new.append(max_list[0]) 
    for i in range(1, len(max_list)-1):
        if ((data[max_list[i]]) < (data[max_list[i+1]]*19/20)) and ((data[max_list[i]])<(data[max_list[i-1]]*19/20)):
            continue
        else:
            max_list_new.append(max_list[i])
    max_list_new.append(max_list[-1]) 
    '''

    return np.array(max_list)


def r_peak_detection(f_ecg_norm):
    from scipy.signal import find_peaks
    r_peaks,_ = find_peaks(f_ecg_norm)

    new_r_peak = list()
    
    for i in range(len(r_peaks)-1):
        if (np.abs(r_peaks[i]-r_peaks[i+1])>100):
            continue
        new_r_peak.append(r_peaks[i])
    

    return np.array(r_peaks)


def self_padd_sigbal_maintaining_heart_rate(ecg_signal, padd_up_to=25, sampling_rate=2000, threshhold=0.4, use_lead=0):

    raw_r_peaks = r_peak_detection(ecg_signal[:, use_lead])

    #detector = Detectors(sampling_rate)
    #r_peaks = detector.christov_detector(ecg_signal[:, use_lead])

    #r_peaks = peak_finding(ecg_signal[:, use_lead], 10, 30

    r_peaks = list()
    for r_peak in raw_r_peaks:
        if ecg_signal[r_peak, use_lead] > threshhold:
            r_peaks.append(r_peak)

    return r_peaks



def self_padd_dataset_maintaining_heart_rate(mat_dataset_dir, padd_up_to=25, sampling_rate=2000, threshhold=0.4, use_lead=0):
    padded_dataset_dir = os.path.join(base_project_dir, 'Data', 'SlefPaddedFixHRMatDataset')
    if not os.path.exists(padded_dataset_dir):
        os.mkdir(padded_dataset_dir)

    signal_names = os.listdir(mat_dataset_dir)
    for signal_name in signal_names:
        ecg_signal_path = os.path.join(mat_dataset_dir, signal_name)

        ecg_signal = loadmat(ecg_signal_path).get('ecg_signal')
        r_peaks = self_padd_sigbal_maintaining_heart_rate(ecg_signal, padd_up_to, sampling_rate, threshhold, use_lead)

        org_n_samples = ecg_signal.shape[0]
        time_axies_org = list(map(lambda x:x/sampling_rate, [i for i in range(org_n_samples)]))

        time_axies_r_peak= list(map(lambda x:x/sampling_rate, [r_peak for r_peak in r_peaks]))

        ecg_at_r_peak = [ecg_signal[peak, use_lead] for peak in r_peaks]
        plt.xlabel('Time(s)')
        plt.ylabel('Padded ECG signal')
        plt.plot(np.arange(len(ecg_signal[:, use_lead])),ecg_signal[:, use_lead])
        plt.scatter(r_peaks, ecg_at_r_peak, marker='o', color='red')
        plt.show()
        #input()
        continue


        '''org_n_samples = ecg_signal.shape[0]
        padded_n_sampled = sampling_rate * padd_up_to
        time_axies_org = list(map(lambda x:x/sampling_rate, [i for i in range(org_n_samples)]))
        time_axies_pad = list(map(lambda x:x/sampling_rate, [i for i in range(padded_n_sampled)]))
        plt.xlabel('Time(s)')
        plt.ylabel('Padded ECG signal')
        plt.plot(time_axies_org, ecg_signal[:, use_lead], time_axies_pad, padded_ecg_signal[:, 0])
        plt.show()
        #input()
        continue'''

        mat_file_path = os.path.join(padded_dataset_dir, signal_name)
        mat_file_content = {'padded_ecg_signal':padded_ecg_signal}



if __name__ =='__main__':
    
    mat_dataset_dir = os.path.join(base_project_dir, 'Data', 'MatPVCVTECGData')

    zero_padd_dataset(mat_dataset_dir, 25, 2000)
    #self_padd_dataset(mat_dataset_dir, 25, 2000)
    #self_padd_dataset_maintaining_heart_rate(mat_dataset_dir, 25, 2000, 2000, 5)


