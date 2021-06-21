# test of classifier
from features import extract_features
import numpy as np
import read_in
from read_in import create_mixed_signal
import os
import pickle

# short piece of code simulating a potential application of the spotter classifier
# reads file and checks window by window if a spotter is needed, if so -> break!


folder = "data/"
good_file = "15_successful_reps.csv"
struggle_file = "15_failed_reps.csv"

good_signal, good_time = read_in.read_in_file(good_file, folder, 20)
struggle_signal, struggle_time = read_in.read_in_file(struggle_file, folder, 20)
mixed_signal = create_mixed_signal(good_signal, struggle_signal, 30, 30)


folder = "training_output"
classifier_file = 'classifier.pickle'
with open(os.path.join(folder, classifier_file), 'rb') as f: # get classifier
    classifier = pickle.load(f)

n = 30
for i in range(0, len(mixed_signal), n): 
    period = np.asarray(mixed_signal[i:i + n])
    period_features = np.reshape(extract_features(period)[1], (1,-1))
    pred = classifier.predict(period_features)
    if(pred == 0):
        print("(!) SPOTTER NEEDED!")
        break

