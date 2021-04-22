'''

BAYES CLASSIFICATION WITH GAUSSIAN CLASS-CONDITIONALS
WITH TOY-GAUSSIANS IN 20D

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
    "font.size": 24})

# %% IMPORT DATA

dataset = 'P2c'
train_data = np.loadtxt('./data/'+dataset+'_train_data_20D.txt', delimiter=',', skiprows=1)
test_data = np.loadtxt('./data/'+dataset+'_test_data_20D.txt', delimiter=',', skiprows=1)

# %% TRAINING

os.makedirs('./models/ex2', exist_ok=True)
path = './models/ex2/'

# SET TRAINING PARAMETERS

num_samples = train_data.shape[0]
training_size = 10
force_train = True

if os.path.isfile(path + 'model_QD_ML_dataset_' + dataset + '_size_' + \
    str(training_size) + '.pkl') and force_train==False:
    
    print('PICKING PRETRAINED MODEL')
    
    f = open(path + 'model_QD_ML_dataset_' + dataset + '_size_' + \
        str(training_size) + '.pkl', 'rb')
    model = pickle.load(f)
    f.close()     

else:
    print('TRAINING IN PROCESS...')

    np.random.seed(562)
    random_idx = np.random.randint(num_samples, size=training_size)

    pos_mean, neg_mean, pos_cov, neg_cov, pos_prior, neg_prior = \
        bayes_tools.train_gaussian_conditionals(train_data[random_idx])

    print('...TRAINING COMPLETE!')

    model = {"means":[pos_mean, neg_mean], "covs":[pos_cov, neg_cov], "priors":[pos_prior, neg_prior]}
    f = open(path + 'model_QD_ML_dataset_' + dataset + '_size_' + str(training_size) + '.pkl', 'wb')
    pickle.dump(model, f)
    f.close()

# %% TESTING

confusion_mtx = bayes_tools.test_gaussian_conditionals(test_data,
    model["means"], model["covs"], model["priors"])

# %% PLOT CONFUSION MATRIX

os.makedirs('./results/ex2', exist_ok=True)
path = './results/ex2/'

save_res = path + 'conf_mtx_QD_ML_dataset_' + dataset + '_size_' + str(training_size)

bayes_tools.plot_confusion_matrix(confusion_mtx, save=save_res)
# %%
