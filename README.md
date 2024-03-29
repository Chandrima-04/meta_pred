# Meta_Pred

Metagenomic data can be coming from multiple sources:
- Targeted sequencing: 16S, 18S, ITS
- Shotgun Sequencing (Illumina, Ultima, etc) 
- Long-Reads (PacBio, ONT, etc) 
- Linked-Reads (10X)

Metagenomics data can be defined as follow:
- Quantitative: Metagenomic results are usually quantitative. Based on the tool they can be absolute or relative values in a given sample
- Categorical: Variables represent types of data which may be divided into groups
- Nominal: Data that is classified without a natural order or rank
- Unbalanced: Most of the dataset are usually unbalanced in nature
- Compositional: Compositional data are quantitative descriptions of the parts of some whole, conveying relative information
- Non-normal and not-Gaussian and not-Bayesian model: Does not follow any distribution pattern
- Sparse Datasets: Most of the value in an matrix are 0
- Usually m>n where m= number of microbes and n=number of samples

It is widely known that metagenomics data has its own signature, but due to variations in data quality, a single tool does not cover all kinds of data. Hence, we developed meta_pred, a metagenomic-based classifier for exploring the metagenomic fingerprint.

![Meta_Pred](https://user-images.githubusercontent.com/9072403/177652386-afea994c-b1f3-41a1-b571-8347da6aca02.jpeg)


The tool offers 6 preprocessing methods and 11 classifiers. It allows the users to choose methods to explore multiple preprocessing methods/cross-validation techniques, to find the best method. 

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

Also, as metagenomic tools are very sensitive, there is a "noisy" option to study the effect of Gaussian noise and stability of the dataset.

## Installation

From source

```bash
git clone https://github.com/Chandrima-04/meta_pred.git
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
General:
meta_pred --help

To find particular mode:
meta_pred [all/one/kfold/leave-one] --help
```

For running meta_pred:
```bash
meta_pred [all/one/kfold/leave-one] <options> [METADATA-FILE] [DATA-FILE] [OUTPUT-FOLDER]
```

#### Options:
```
  --test-size FLOAT           The relative size of the test data
  --num-estimators INTEGER    Number of trees in our Ensemble Methods
  --num-neighbours INTEGER    Number of clusters in our knn
  --model-name TEXT           The model type to train
  --normalize-method TEXT     Normalization method
  --feature-name TEXT         The feature to predict
  --normalize-threshold TEXT  Normalization threshold for binary normalization
  --noisy BOOLEAN             Whether add Gaussian noise or not
  metadata_file CSV_FILE      CSV file with metadata, features defining the location of data collection
  data_file CSV_FILE          CSV file with samples as row and microbes associated as column 
  out_folder TEXT             Name of the output folder to be created
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

Note: For the purpose of simulating noise in the natural environment (sequencing, collection bias), we also added Gaussian noise to gauge the impact on the models’ performance.
![Gaussian_noise](https://user-images.githubusercontent.com/9072403/177815688-cee08db7-22fe-4956-929b-c2c647dd5b03.png)


### ML Expertise Mode (Have your own Fav model)
one: When you already have a choosen classifier and preprocessing method.

```bash
meta_pred one --model-name linear_svc --normalize-method binary  --feature-name city toy_data/toy_metadata.csv toy_data/toy_input.csv toy_data/toy_one
```

#### Output

- Confusion matrix
- CSV file consisting of report named on model selected

```bash
head toy_data/toy_one/linear_svc_binary.csv 

,Accuracy,Precision,Recall
linear_svc binary,0.8955223880597015,0.8955223880597015,0.8955223880597015
```

### Whiz Mode (validating using cross-validation!)

kfold: Machine Learning based cross-validation, with k=10 (Can be modified by user).

```bash
meta_pred kfold --k-fold 5 --normalize-method clr --feature-name continent toy_data/toy_metadata.csv toy_data/toy_input.csv toy_data/toy_kfold
```

#### Output
- CSV file consisting of report named on model selected

```bash
head toy_data/toy_kfold/random_forest_clr.csv

,Best Score,Mean Score,Standard Deviation
random_forest clr,0.9056603773584906,0.8785195936139333,0.03640502469381202
```

### Explorer Mode (validating using LOGO cross-validation!)

leave-one: Leave One Group Out cross-validation which provides train/test indices to split data according to a third-party provided group. This group information can be used to encode arbitrary domain specific stratifications of the samples as integers. The parameters *group-name* (usually arbitary group) and *feature-name* (usually location like city, continent) needs to be set.


## Citation
Bhattacharya, C., Tierney, B.T., Ryon, K.A., Bhattacharyya, M., Hastings, J.J., Basu, S., Bhattacharya, B., Bagchi, D., Mukherjee, S., Wang, L. and Henaff, E.M., 2022. Supervised Machine Learning Enables Geospatial Microbial Provenance. Genes, 13(10), p.1914.

## Datasets

The datasets used in this Paper are from 

- MetaSUB: Danko, D., Bezdan, D., Afshin, E.E., Ahsanuddin, S., Bhattacharya, C., Butler, D.J., Chng, K.R., Donnellan, D., Hecht, J., Jackson, K. and Kuchin, K., 2021. A global metagenomic map of urban microbiomes and antimicrobial resistance. Cell, 184(13), pp.3376-3393.

- TARA Ocean: Salazar, G., Paoli, L., Alberti, A., Huerta-Cepas, J., Ruscheweyh, H.J., Cuenca, M., Field, C.M., Coelho, L.P., Cruaud, C., Engelen, S. and Gregory, A.C., 2019. Gene expression changes and community turnover differentially shape the global ocean metatranscriptome. Cell, 179(5), pp.1068-1083.



## License

All material is provided under the MIT License.

## Credits

This package is written and maintained by [Chandrima Bhattacharya](mailto:chb4004@med.cornell.edu).
