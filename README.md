# Effet Oppos

The program helps to predict metagenomic signature of a city. It can be trained to use multiple classifier algorithm to predict the fingerprint.
Following preprocessing can be done:
raw: no preprocessing 
total-sum: having the sum of each row to be 1
standard scalar: forces the column to have a mean of 0
binary: 0,1 based on a threshold
Also, multiple classifier can be used to train the model, including Random Forest, KNN, Gaussian, Linear Support Vector Machine, Support Vector Machine and Multi-layer Perceptron. 

## Installation

From source
```
git clone git@github.com:Chandrima-04/MetaSUB.git
cd MetaSUB
python setup.py install
```

To run tests
```
cd path/to/effet_oppos
python -m pytest --color=yes . -s
```

## Credits

This package is written and maintained by [Chandrima Bhattacharya](mailto:chb4004@med.cornell.edu).
