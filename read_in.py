import csv
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, freqz, filtfilt, firwin, iirnotch, lfilter

# read in data

def read_in_file(file_name, dir, omit_value):
    data_file = os.path.join(dir, file_name)
    signal = []
    time = []
    with open(data_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if(float(row[3]) < omit_value):
                continue
            x_y_z = np.array( [float(row[0]), float(row[1]), float(row[2]), int(row[4])] )
            signal.append(x_y_z)
            time.append(float(row[3]))

    return np.array(signal), np.array(time)

def get_mag_signal(data):
    mag_signal = []
    for row in data:
        mag_signal.append( math.sqrt(pow(row[0], 2) + pow(row[1], 2) + pow(row[2], 2)) )
    return mag_signal

def get_y_signal(data):
    y_signal = []
    for row in data:
        y_signal.append(row[1])
    return y_signal

def graph_data(data, time):
    mag_signal = get_mag_signal(data)
    cleaned_sgnal = clean_signal(mag_signal)
    plt.plot(time, cleaned_sgnal, label = "magnitude_signal")
    plt.show()

def clean_signal(signal):
    order = 5
    fs = 60  # sample rate, Hz
    cutoff = 2  # desired cutoff frequency of the filter, Hz. MODIFY AS APPROPROATE
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def create_mixed_signal(good_signal, struggle_signal, good_index, struggle_index): # creates a mixed signal
    mixed_signal = []
    for i in range(0, good_index):
        mixed_signal.append(good_signal[i])
    for i in range(0, struggle_index):
        mixed_signal.append(struggle_signal[i])
    return mixed_signal

def graph_both(signal_one, signal_two, time):
    plt.plot(time[:326], signal_one[:326], label = "successful reps")
    plt.plot(time[:326], signal_two[:326], label = "failed reps")

    plt.show()
