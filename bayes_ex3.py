'''

BAYES CLASSIFICATION WITH GAUSSIAN CLASS-CONDITIONALS
WITH TOY-GAUSSIANS IN 1D

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

dataset = 'P3a'
train_data = np.loadtxt('./data/'+dataset+'_train_data.txt', skiprows=1)
test_data = np.loadtxt('./data/'+dataset+'_test_data.txt', skiprows=1)

# %% TRAINING

os.makedirs('./models/ex3', exist_ok=True)
path = './models/ex3/'

# SET TRAINING PARAMETERS
np.random.seed(34)

num_samples = train_data.shape[0]
training_size = num_samples
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

# %% PLOT SAMPLES

os.makedirs('./results/ex3', exist_ok=True)
path = './results/ex3/'
save_res = path + 'samples_QD_ML_dataset_' + dataset + '_size_' + str(training_size)

plt.figure(figsize=(8,8))
ax = plt.gca()

bayes_tools.plot_data1D(train_data, ax=ax, xlimits=[-18,18], show=False)
bayes_tools.plot_gaussian(model["means"][0], model["covs"][0], ax=ax,
    colour='red', xlimits=[-10,18], ylimits=[-.1,0.2], show=False)
bayes_tools.plot_gaussian(model["means"][1], model["covs"][1], ax=ax,
    colour='green', xlimits=[-10,18], ylimits=[-.1,0.2], show=False)
bayes_tools.plot_decisionboundary1D(model["means"], model["covs"],
    model["priors"], ax=ax, xlimits=[-10,18], num_points=500,
    show=True, save=save_res)

# %% TESTING

confusion_mtx = bayes_tools.test_gaussian_conditionals(test_data,
    model["means"], model["covs"], model["priors"])

# %% PLOT CONFUSION MATRIX

save_res = path + 'conf_mtx_QD_ML_dataset_' + dataset + '_size_' + str(training_size)

bayes_tools.plot_confusion_matrix(confusion_mtx, save=save_res)
