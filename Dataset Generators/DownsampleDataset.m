clc;
clear;
clear all;
close all;
clc;

addpath(strcat(pwd, '\..\Filters'))
source_dir = strcat(pwd, '\..\Data\SlefPaddedMatDataset\');
dist_dir = strcat(pwd, '\..\Data\SelfPaddedFiltered50HzDataset\');

sampling_rate = 2000;

mkdir(dist_dir)
    
files = dir(source_dir);
for file = files'
    if file.isdir ~= 1
        ecg_signal_file_path = strcat(source_dir, '\', file.name);
        ecg_signal = importdata(ecg_signal_file_path);
      
        ecg_signal_size = size(ecg_signal);
        
        n_samples = ecg_signal_size(1);

        filtered_signal = zeros(n_samples, 12);
        for i = 1:12
            filtered_signal(:, i) = lowpass(ecg_signal(:, i), 25, sampling_rate);
        end
        
        filtered_signal_50hz = downsample(filtered_signal, 40);
        
        %filtered_signal_50hz_size = size(filtered_signal_50hz)
        %n_samples_down = filtered_signal_50hz_size(1)
   
        %time_axies_2000hz = (1:n_samples)/sampling_rate;
        %time_axies_50hz = (1:n_samples_down)/50;
        
        %figure;
        %plot(time_axies_2000hz, ecg_signal(:, 1), time_axies_50hz, filtered_signal_50hz(:, 1))
        %title('Original vs Preprocessed ECG Signals')
        %legend('Original Signal','Filtered & Downsampled Signal')
        %pause;

        downsampled_path = strcat(dist_dir, '\', file.name);
        save(downsampled_path, 'filtered_signal_50hz');

        %fprintf('\n\n\n\n\n')

    end
end

source_dir = strcat(pwd, '\..\Data\ZeroPaddedMatDataset\');
dist_dir = strcat(pwd, '\..\Data\ZeroPaddedFiltered50HzDataset\');

mkdir(dist_dir)
    
files = dir(source_dir);
for file = files'
    if file.isdir ~= 1
        ecg_signal_file_path = strcat(source_dir, '\', file.name);
        ecg_signal = importdata(ecg_signal_file_path);
      
        ecg_signal_size = size(ecg_signal);
        
        n_samples = ecg_signal_size(1);

        filtered_signal = zeros(n_samples, 12);
        for i = 1:12
            filtered_signal(:, i) = lowpass(ecg_signal(:, i), 25, sampling_rate);
        end
        
        filtered_signal_50hz = downsample(filtered_signal, 40);
        
        %filtered_signal_50hz_size = size(filtered_signal_50hz)
        %n_samples_down = filtered_signal_50hz_size(1)
   
        %time_axies_2000hz = (1:n_samples)/sampling_rate;
        %time_axies_50hz = (1:n_samples_down)/50;
        
        %figure;
        %plot(time_axies_2000hz, ecg_signal(:, 1), time_axies_50hz, filtered_signal_50hz(:, 1))
        %title('Original vs Preprocessed ECG Signals')
        %legend('Original Signal','Filtered & Downsampled Signal')
        %pause;

        downsampled_path = strcat(dist_dir, '\', file.name);
        save(downsampled_path, 'filtered_signal_50hz');

        %fprintf('\n\n\n\n\n')

    end
end




