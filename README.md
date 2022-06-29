# Meta_Pred

Metagenomic data can be coming from multiple sources:
- Targeted sequencing: 16S, 18S, ITS
- Whole Genome Sequencing: both Shotgun (Illumina) or Long-Reads (PacBio or Nanopore) or even Linked-Reads (10X)

Metagenomics data can be defined as follow:
- Quantatitive: Metagenomic results are usually quantative. Based on the tool they can be absolute or relative values in a given sample.
- Categorical: Variables represent types of data which may be divided into groups
- Nominal: Data that is classified without a natural order or rank
- Unbalanced: Most of the dataset are usually unbalanced in nature
- Compositional: Compositional data are quantitative descriptions of the parts of some whole, conveying relative information
- Non-normal and not-Gaussian and not-Bayesian model: Does not follow any distribution pattern
- Sparse Datasets: Most of the value in an matrix are 0
- Usually m>n where m= number of microbes and n=number of samples

It is known widely that metagenomics data has its own signature, but due to the variation in data quality, a single tool does not cover all kind of data. Hence, we develop meta_pred, a metagenomic-based classifier for exploring metagenomic fingerprint.

The tool offers 6 preprocessing methods and 12 classifiers. It allows the users to choose methods to explore multiple preprocessing methods/cross-validation techniques, to find the best method. 


## Features

### Preprocessing

Following preprocessing can be done:

- binary: 0,1 based on a threshold value (default=0.0001)
- clr: transformation function for compositional data based on Aitchison geometry to the real space
- multiplicative_replacement: transformation function for compositional data  uses the multiplicative replacement strategy for replacing zeros such that compositions still add up to 1
- raw: no preprocessing
- standard scalar: forces the column to have a mean of 0
- total-sum: relative abundance method having the sum of each row to be 1

NOTE: Multiple methods of transformation/preprocessing will distort the data.

### Classification

Also, multiple classifiers can be used to train the model, including:

- Tree Based Methods: Decision Tree Classifier
- Bagging (Bootstrap Aggregation): Random Forest Classifier and ExtraTree
- Boosting: AdaBoost
- k-Nearest Neighbour
- Bayesian Classifier: NBC(gaussianNB) 
- SVM: Linear Support Vector Machine and Support Vector Machine
- Regression Methods: Logistic Regression and Linear Discriminant Analysis 
- Voting Classifier: Consisting of top performing model based on Linear SVM, Random Forest and Logistic Regression


### Cross-validation

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

For help:
```bash
meta_pred one --help
```

For running meta_pred:
```bash
meta_pred [all/one/kfold/leave-one] <options> [METADATA-FILE] [DATA-FILE] [OUTPUT-FOLDER]
```

The different modes are:
all: To evaluate all 12 classifier methods along with 6 preprocessing. Usually comes with additional *noisy* parameter which adds a Gaussian noise between 0.0000000001-1000 to test the tolerance of the data.

one: When you already have a choosen classifier and preprocessing method.

kfold: Machine Learning based cross-validation, with k=10 (Can be modified by user).

leave-one: Leave One Group Out cross-validation which provides train/test indices to split data according to a third-party provided group. This group information can be used to encode arbitrary domain specific stratifications of the samples as integers. The parameters *group-name* (usually arbitary group) and *feature-name* (usually location like city, continent) needs to be set.


```bash
Options:
  --test-size FLOAT           The relative size of the test data
  --num-estimators INTEGER    Number of trees in our Ensemble Methods
  --num-neighbours INTEGER    Number of clusters in our knn
  --model-name TEXT           The model type to train
  --normalize-method TEXT     Normalization method
  --feature-name TEXT         The feature to predict
  --normalize-threshold TEXT  Normalization threshold for binary
                              normalization.
  --model-filename TEXT       Filename of previously saved model
```

## Output

NOTE: Precision, recall, accuracy are calculated based on micro average assuming unbalanced classes in dataset.

#### all/one:

```bash
-- File consisting of accuracy, precision, recall, and other parameters.
-- Folder with all confusion matrix
```

#### kfold/loo:

```bash
-- File consisting of best score, mean score and standard deviation.
```

## License

All material is provided under the MIT License.

## Credits

This package is written and maintained by [Chandrima Bhattacharya](mailto:chb4004@med.cornell.edu).
