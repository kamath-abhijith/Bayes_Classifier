'''

BAYES CLASSIFICATION USING GAUSSIAN
MIXTURE MODEL ESTIMATION IN 2D

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

dataset = 'P1b'
train_data = np.loadtxt('./data/'+dataset+'_train_data_2D.txt', delimiter=',', skiprows=1)
test_data = np.loadtxt('./data/'+dataset+'_test_data_2D.txt', delimiter=',', skiprows=1)

# %% TRAIN GMM

os.makedirs('./models/ex1', exist_ok=True)
path = './models/ex1/'

# SET TRAINING PARAMETERS

num_samples = train_data.shape[0]
training_size = num_samples
force_train = True

if os.path.isfile(path + 'model_GMM_EM_dataset_' + dataset + '_size_' + \
    str(training_size) + '.pkl') and force_train==False:
    
    print('PICKING PRETRAINED MODEL')
    
    f = open(path + 'model_GMM_EM_dataset_' + dataset + '_size_' + \
        str(training_size) + '.pkl', 'rb')
    model = pickle.load(f)
    f.close()

else:
    print('TRAINING IN PROGRESS...')

    priors, means, covs, cost = bayes_tools.train_gmm(train_data, num_components=2,
        max_iter=200, tol=1e-12)

    print('...TRAINING COMPLETE!')

    model = {"means":means, "covs":covs, "priors":priors}
    f = open(path + 'model_GMM_EM_dataset_' + dataset + '_size_' + str(training_size) + '.pkl', 'wb')
    pickle.dump(model, f)
    f.close()

# %% PLOT

os.makedirs('./results/ex1', exist_ok=True)
path = './results/ex1/'

# PLOT LIKELIHOOD AS LOSS
fig = plt.figure(figsize=(8,8))
ax = plt.gca()

save_res = path + 'loss_GMM_EM_dataset_' + dataset + '_size_' + str(training_size)

bayes_tools.plot_loss(cost, ax=ax, \
    yaxis_label=r'$\ln f(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$',
    save=save_res)

# PLOT SAMPLES
save_res = path + 'samples_GMM_EM_dataset_' + dataset + '_size_' + str(training_size)

fig = plt.figure(figsize=(8,8))
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

save_res = path + 'conf_mtx_GMM_EM_dataset_' + dataset + '_size_' + str(training_size)

bayes_tools.plot_confusion_matrix(confusion_mtx, show=True, save=save_res)
# %%
