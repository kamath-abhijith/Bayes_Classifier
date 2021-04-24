# BAYES CLASSIFIER

Bayes' Classifier is an optimal multiclass supervised classification method. Repository contains Python3 scripts for two-class classification on:

- toy-Gaussian data in 1D trained using MLE with Gaussian class conditional model,
- toy-Gaussian data in 2D trained using MLE with Gaussian class conditional model,
- toy-Gaussian data in 2D trained using MLE with Exponential class conditional model,
- toy-Gaussian data in 2D trained using EM with Gaussian Mixture Model,
- toy-Gaussian data in 20D trained using MLE with Gaussian distributed data,
- text corpus trained using naive Bayes' classifier with BoG model and TF-IDF features.

## Documentation

`docs/instructions.pdf` contains the necessary instructions for the assignment, and `docs/solutions.pdf` contains the results and inferences.

## Installation

Clone this repository and install the requirments using
```shell
https://github.com/kamath-abhijith/Bayes_Classifier
conda create --name <env> --file requirements.txt
```

## Run

- `/data/` contains the data files for the experiments. Description of the data are included in `docs/solutions.pdf`
- Run `bayes_ex(x).py` to run Bayes' classifier for exercise `(x)`. Change the `training_size` and `dataset` variables.
- Run `nn_ex(x).py` to run nearest-neighbour classifier for exercise `(x)`. Change the `training_size` and `dataset` variables.
- Run `gmm_ex(x).py` to run Bayes' classifier with Gaussian mixture model for exercise `(x)`. Change the `training_size` and `dataset` variables.
- Run `naiveBayes_doc_class.py` to run sentiment analysis using naive Bayes' classifier.
- Find the results in `results` and the saved models in `models`.