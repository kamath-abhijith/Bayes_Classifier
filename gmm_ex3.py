'''

BAYES CLASSIFICATION USING GAUSSIAN
MIXTURE MODEL ESTIMATION IN 1D

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

dataset = 'P3b'
train_data = np.loadtxt('./data/'+dataset+'_train_data.txt', skiprows=1)
test_data = np.loadtxt('./data/'+dataset+'_test_data.txt', skiprows=1)

# %% TRAIN GMM

os.makedirs('./models/ex3', exist_ok=True)
path = './models/ex3/'

# SET TRAINING PARAMETERS
np.random.seed(34)

num_samples = train_data.shape[0]
training_size = num_samples
force_train = False

if os.path.isfile(path + 'model_GMM_EM_dataset_' + dataset + '_size_' + \
    str(training_size) + '.pkl') and force_train==False:
    
    print('PICKING PRETRAINED MODEL')
    
    f = open(path + 'model_GMM_EM_dataset_' + dataset + '_size_' + \
        str(training_size) + '.pkl', 'rb')
    model = pickle.load(f)
    f.close()

else:
    print('TRAINING IN PROGRESS...')

    dim = train_data.shape[1]
    labels = train_data[:,dim-1]

    num_classes = len(np.unique(labels))
    components = 2

    means = np.zeros((num_classes, components, dim-1))
    covs = np.zeros((num_classes, components, dim-1, dim-1))
    priors = np.zeros((num_classes, components))

    for classes in range(num_classes):

        data = train_data[np.where(labels==np.unique(labels)[classes])]
        num_data = data.shape[0]

        random_idx = np.random.randint(num_data, size=training_size)
        priors[classes], means[classes], covs[classes], _ = \
            bayes_tools.train_gmm(data[random_idx], num_components=components,\
                max_iter=200, tol=1e-36)

    print('...TRAINING COMPLETE!')

    model = {"means":means, "covs":covs, "priors":priors}
    f = open(path + 'model_GMM_EM_dataset_' + dataset + '_size_' + str(training_size) + '.pkl', 'wb')
    pickle.dump(model, f)
    f.close()

# %% PLOT DISTRIBUTIONS

os.makedirs('./results/ex3', exist_ok=True)
path = './results/ex3/'

fig = plt.figure(figsize=(8,8))
ax = plt.gca()
colours = ['green', 'red']

bayes_tools.plot_data1D(train_data, ax=ax, xlimits=[-18,18], show=False)

num_classes = model["means"].shape[0]
num_components = model["means"].shape[1]
for classes in range(num_classes):
    for component in range(num_components):
        bayes_tools.plot_gaussian(model["means"][classes][component],
            model["covs"][classes][component], ax=ax, colour=colours[classes],
            xlimits=[-10,18], ylimits=[-.1,0.3], show=False)

save_res = path + 'samples_GMM_EM_dataset_' + dataset + '_size_' + str(training_size)
plt.savefig(save_res + '.pdf', format='pdf')

# %% TESTING

confusion_mtx = bayes_tools.test_gmm(test_data,
    model["means"], model["covs"], model["priors"])

# %% PLOT CONFUSION MATRIX

save_res = path + 'conf_mtx_GMM_dataset_' + dataset + '_size_' + str(training_size)

bayes_tools.plot_confusion_matrix(confusion_mtx, save=save_res)