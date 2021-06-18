# Meta_Pred

The program helps to predict metagenomic signature of an environment. An microbial environment can include a variety of ecologies (extreme environment, build environment, natural environment, etc) as well as host environment (human, animal, fungi,  plant, etc). It can be trained to use multiple classifier algorithm to predict the fingerprint.

Following preprocessing can be done:
- raw: no preprocessing ;
- total-sum: having the sum of each row to be 1;
- standard scalar: forces the column to have a mean of 0;
- binary: 0,1 based on a threshold.

Also, multiple classifier can be used to train the model, including:
- Tree Based Methods: Decision Tree Classifier
- Bagging (Bootstrap Aggregation): Random Forest Classifier and ExtraTree
- Boosting: AdaBoost, CatBoost and LightGBM
- k-Nearest Neighbour
- Bayesian Classifier: NBC(gaussianNB) and Gaussian
- SVM - Linear Support Vector Machine and Support Vector Machine
- Nueral Network - Multi-layer Perceptron
- Regression Methods - Logistic Regression, PLSR and Linear Discriminant Analysis  

The different types of Cross-Validations included are:
- normal cross-validation 
- kfold cross-validation
- leave one group out cross-validation

The split size can be manually modified. Default is 80:20 train:test cross-validation.

Also, as metagenomic tools are very sensitive, there is an "noisy" option to study the effect of Gaussian noise and stability of the dataset.

## Installation

From source
```
git clone https://github.com/Chandrima-04/meta_pred.git
cd meta_pred
python setup.py install
```

To run tests
```
cd path/to/meta_pred
python -m pytest --color=yes . -s
```

## Usage

```
meta_pred [all/one/k-fold/leave-one] <options> [METADATA-FILE] [DATA-FILE] [OUTPUT-FOLDER]
```

## Credits

This package is written and maintained by [Chandrima Bhattacharya](mailto:chb4004@med.cornell.edu).
