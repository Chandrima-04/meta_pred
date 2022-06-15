# Meta_Pred

Metagenomic data can be coming from multiple sources:
- Targeted sequencing: 16S, 18S, ITS
- Whole Genome Sequencing: both Shotgun (Illumina) or Long-Reads (PacBio or Nanopore) or even Linked-Reads (10X)



Meta_pred is a computational prediction based tool for exploring metagenomic fingerprint or signature. The tool offers multiple modules, including user selection of model, multiple cross-validation for model stability, and can be applied for all kind of metagenomic data, including OTUs, WGS, long reads. 

## Features

### Preprocessing

Following preprocessing can be done:

- raw: no preprocessing
- total-sum: relative abundance method having the sum of each row to be 1
- standard scalar: forces the column to have a mean of 0
- binary: 0,1 based on a threshold

### Classification

Also, multiple classifiers can be used to train the model, including:

- Tree Based Methods: Decision Tree Classifier
- Bagging (Bootstrap Aggregation): Random Forest Classifier and ExtraTree
- Boosting: AdaBoost, CatBoost and LightGBM
- k-Nearest Neighbour
- Bayesian Classifier: NBC(gaussianNB) and Gaussian
- SVM - Linear Support Vector Machine and Support Vector Machine
- Nueral Network - Multi-layer Perceptron
- Regression Methods - Logistic Regression and Linear Discriminant Analysis  

The different types of Cross-Validations included are:

- normal cross-validation
- kfold cross-validation
- leave one group out cross-validation

The split size can be manually modified. Default is 80:20 train:test cross-validation.

Also, as metagenomic tools are very sensitive, there is an "noisy" option to study the effect of Gaussian noise and stability of the dataset.

## Installation

From source

```bash
git clone git@github.com:Chandrima-04/meta_pred.git
cd meta_pred
python setup.py install
```

## Usage

```bash
meta_pred [all/one/k-fold/leave-one] <options> [METADATA-FILE] [DATA-FILE] [OUTPUT-FOLDER]
```

Run 'meta_pred one --help' for help.

```bash
Options:
  --test-size FLOAT           The relative size of the test data
  --num-estimators INTEGER    Number of trees in our Ensemble Methods
  --num-neighbours INTEGER    Number of clusters in our knn/MLknn
  --n-components INTEGER      Number of components for dimensionality
                              reduction in Linear Discriminant Analysis
  --model-name TEXT           The model type to train
  --normalize-method TEXT     Normalization method
  --feature-name TEXT         The feature to predict
  --normalize-threshold TEXT  Normalization threshold for binary
                              normalization.
  --model-filename TEXT       Filename of previously saved model
```

## Credits

This package is written and maintained by [Chandrima Bhattacharya](mailto:chb4004@med.cornell.edu).
