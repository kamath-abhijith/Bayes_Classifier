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

# %%
