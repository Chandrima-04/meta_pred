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

![Meta_Pred](https://user-images.githubusercontent.com/9072403/177652386-afea994c-b1f3-41a1-b571-8347da6aca02.jpeg)


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

## Requirements

- Python >= 3.6
- Cython
- scikit-learn
- scikit-bio
- numpy
- pandas
- scipy
- click

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

### Newbie Mode (What is Machine Learning?)
all: To evaluate all 12 classifier methods along with 6 preprocessing. Usually comes with additional *noisy* parameter which adds a Gaussian noise between 0.0000000001-1000 to test the tolerance of the data.

```bash
meta_pred all --noisy TRUE --feature-name city toy_data/toy_metadata.csv toy_data/toy_input.csv toy_data/toy_all
```

#### Output

The file consists of 
- Directory consisting of confusion matrix for each run
- Directory consisting of predicted confusion matrix for each run with annotated feature names
- Summary file called output_metrics.csv 

```bash
head toy_data/toy_all/output_metrics.csv 

,Classifier,Preprocessing,Noise,Training_Time_in_sec,Accuracy,Top_2_accuracy,Top_3_accuracy,Top_5_accuracy,Top_10_accuracy,Precision,Recall
adaboost_binary_0,adaboost,binary,0,25.02643466,0.611111111,0.777777778,0.805555556,0.944444444,1,0.611111111,0.611111111
adaboost_binary_1e-10,adaboost,binary,1.00E-10,117.9060383,0.611111111,0.777777778,0.833333333,0.916666667,1,0.611111111,0.611111111
adaboost_binary_1e-09,adaboost,binary,1.00E-09,123.2134194,0.666666667,0.777777778,0.888888889,0.944444444,1,0.666666667,0.666666667
adaboost_binary_1e-08,adaboost,binary,1.00E-08,122.9907653,0.666666667,0.694444444,0.833333333,0.944444444,1,0.666666667,0.666666667
adaboost_binary_1e-07,adaboost,binary,1.00E-07,131.499526,0.555555556,0.777777778,0.833333333,0.944444444,1,0.555555556,0.555555556
adaboost_binary_1e-06,adaboost,binary,1.00E-06,149.7495966,0.277777778,0.472222222,0.722222222,0.888888889,1,0.277777778,0.277777778
adaboost_binary_1e-05,adaboost,binary,1.00E-05,206.7691383,0.333333333,0.5,0.583333333,0.833333333,1,0.333333333,0.333333333
adaboost_binary_0.0001,adaboost,binary,0.0001,219.425483,0.25,0.416666667,0.5,0.861111111,1,0.25,0.25
adaboost_binary_0.001,adaboost,binary,0.001,214.7230728,0.222222222,0.305555556,0.444444444,0.722222222,1,0.222222222,0.222222222

```

one: When you already have a choosen classifier and preprocessing method.

kfold: Machine Learning based cross-validation, with k=10 (Can be modified by user).

leave-one: Leave One Group Out cross-validation which provides train/test indices to split data according to a third-party provided group. This group information can be used to encode arbitrary domain specific stratifications of the samples as integers. The parameters *group-name* (usually arbitary group) and *feature-name* (usually location like city, continent) needs to be set.


```
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
