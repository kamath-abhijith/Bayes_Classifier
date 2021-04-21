'''

NEAREST-NEIGHBOUR CLASSIFICATION

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES
import os
import pickle
import numpy as np

from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

import bayes_tools

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 11})

# %% IMPORT DATA

dataset = 'P1b'
train_data = np.loadtxt('./data/'+dataset+'_train_data_2D.txt', delimiter=',', skiprows=1)
test_data = np.loadtxt('./data/'+dataset+'_test_data_2D.txt', delimiter=',', skiprows=1)

# %% TESTING

num_samples = train_data.shape[0]
training_size = 75
random_idx = np.random.randint(num_samples, size=training_size)

order = 1

confusion_mtx = bayes_tools.test_knn(train_data[random_idx], test_data,
    metric='euclidean', order=order)

# %% PLOT CONFUSION MATRIX

os.makedirs('./results/ex1', exist_ok=True)
path = './results/ex1/'

save_res = path + 'conf_mtx_KNN_dataset_' + dataset + '_size_' + str(training_size)

bayes_tools.plot_confusion_matrix(confusion_mtx, save=save_res)