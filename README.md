# Classifiers

The program helps to predict metagenomic signature of a city. It can be trained to use multiple classifier algorithm to predict the fingerprint.
Following preprocessing can be done:
- raw: no preprocessing ;
- total-sum: having the sum of each row to be 1;
- standard scalar: forces the column to have a mean of 0;
- binary: 0,1 based on a threshold.

Also, multiple classifier can be used to train the model, including
- Tree Based Methods: Decision Tree Classifier
- Bagging (Bootstrap Aggregation): Random Forest Classifier and ExtraTree
- Boosting: AdaBoost
- k-Nearest Neighbour
- Bayesian Classifier: NBC(gaussianNB) and Gaussian
- SVM - Linear Support Vector Machine and Support Vector Machine
- Nueral Network - Multi-layer Perceptron
- Regression Methods - Linear Discriminant Analysis  

## Installation

From source
```
git clone git@github.com:Chandrima-04/meta_pred.git
cd meta_pred
python setup.py install
```

To run tests
```
cd path/to/effet_oppos
python -m pytest --color=yes . -s
```

## Credits

This package is written and maintained by [Chandrima Bhattacharya](mailto:chb4004@med.cornell.edu).
