'''

BAYES CLASSIFICATION WITH GAUSSIAN CLASS-CONDITIONALS
WITH TOY-GAUSSIANS IN 2D

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

dataset = 'P1c'
train_data = np.loadtxt('./data/'+dataset+'_train_data_2D.txt', delimiter=',', skiprows=1)
test_data = np.loadtxt('./data/'+dataset+'_test_data_2D.txt', delimiter=',', skiprows=1)

# %% TRAINING

os.makedirs('./models/ex1', exist_ok=True)
path = './models/ex1/'

# SET TRAINING PARAMETERS
np.random.seed(34)

num_samples = train_data.shape[0]
training_size = 75
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

    random_idx = np.random.randint(num_samples, size=training_size)

    pos_mean, neg_mean, pos_cov, neg_cov = \
        bayes_tools.train_gaussian_conditionals(train_data[random_idx])

    print('...TRAINING COMPLETE!')

    model = {"means":[pos_mean, neg_mean], "covs":[pos_cov, neg_cov], "priors":[0.5, 0.5]}
    f = open(path + 'model_QD_ML_dataset_' + dataset + '_size_' + str(training_size) + '.pkl', 'wb')
    pickle.dump(model, f)
    f.close()

# %% PLOT DISTRIBUTIONS

os.makedirs('./results/ex1', exist_ok=True)
path = './results/ex1/'
save_res = path + 'samples_QD_ML_dataset_' + dataset + '_size_' + str(training_size)

plt.figure(figsize=(8,8))
ax = plt.gca()

bayes_tools.plot_data2D(train_data, ax=ax, xlimits=[-4,10],
    ylimits=[-4,10], show=False)
bayes_tools.plot_confidence_ellipse2D(model["means"][0], model["covs"][0],
    nstd=3, ax=ax, colour='red')
bayes_tools.plot_confidence_ellipse2D(model["means"][1], model["covs"][1],
    nstd=3, ax=ax, colour='green')
bayes_tools.plot_decisionboundary2D(model["means"], model["covs"],
    model["priors"], ax=ax, num_points=500, show=True, save=save_res)

# %% TESTING

confusion_mtx = bayes_tools.test_gaussian_conditionals(test_data,
    model["means"], model["covs"], model["priors"])

# %% PLOT CONFUSION MATRIX

save_res = path + 'conf_mtx_QD_ML_dataset_' + dataset + '_size_' + str(training_size)

bayes_tools.plot_confusion_matrix(confusion_mtx, save=save_res)

# %%
