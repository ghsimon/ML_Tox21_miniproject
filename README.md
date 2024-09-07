# ML mini project on toxicity classification for the Tox21 dataset

The project was conducted in context of the course "Applied machine learning in chemistry SS 24" instructed by Prof. Ariane Ferreira Nunes Alves (TU Berlin).

## Dataset

The **Tox21 data set** comprises of a large number of toxicity measurements for 12 biological targets across a wide range of chemical compounds. For each target, qualitative toxicity assessments are provided, categorizing chemical compounds as either non-toxic (label 0) or toxic (label 1).

The dataset can therefore be considered a collection of twelve parallel datasets, each corresponding to a biological target. For each target, models can be constructed aiming to predict the toxicity (categorical label) of a chemical compound based on its molecular descriptors.

In this mini project, toxicity data for the Nuclear Receptor: Androgen Receptor (NR-AR) is investigated and processed for later model building. The androgen receptor is a protein that is activated by binding an androgenic harmone (such as testosterone) in the cytoplasm of the cell and subsequently translocates to the nucleus, where it binds DNA as a transcription factor to regulate gene expression. 

Link to the [paper](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a).

Dataset [website]{https://moleculenet.org/datasets-1} and [download](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz).


## Data Preprocessing

The dataset is downloaded and preprocessed in the `data_preprocessing.ipynb` notebook. Molecular features are calculated for all 7265 compounds for which NR-AR toxicity labels using RDKit by interpreting their SMILES codes. The notebook includes a prelimineray analysis of the data including computation of a correlation matrix of the molecular features. Subsequently, features with very low variance AND very low correlation to the toxicity label are removed, reducing feature space from 194 features to 153 features. The data is split in an 80/20 training/test set using stratified splitting.

## k-Nearest Neighbours

The k-nearest neighbours (kNN) algorithm is used as a baseline model in full feature space in the `kNN.ipynb` notebook. The effect of the amount of neighbours $k$ is studied with regards to performance metrics including accuracy, sensitivity, precision, f1-score, ROC-AUC and PR-AUC. Models achieved high accuracy (0.97), reasonable precision (0.81) but low sensitivity (0.4) and PR-AUC (0.48) in the test data.

## Logistic Regression

Logistic regression in full feature space including hyperparameter tuning and cross-validation is performed in the `LogReg.ipynb` notebook. Performance metrics are very close to the baseline model. Choice of parameters (regularization) only has minimal effect on the model performance.

Modelling, hyperparameter tuning and cross-validation are redone in the `LogReg_balanced.ipynb` notebook while weighting regression for the class frequency to account for the class imbalance using the keyword `class_weight='balanced'`. While the model has improved sensitivity, the accuracy and precision decrease significantly. Arguably, this is a safer model for toxicity classification since the cost of false negatives is expected to be very high.

## Random Forest Classifier

Random Forest Classifier models are built in full feature space including hyperparameter tuning and cross-validation in the `RandForrClass.ipynb` notebook. Performance metrics are very close to the baseline model, with slightly improved sensitivity and PR-AUC. The keyword `class_weight='balanced'` was included in the hyperparameter tuning. Choice of hyperparameters only has minimal effect on the model performance. Hyperparameter tuning was repeated optimizing PR-AUC instead of f1 in the `RandForrClass_OptPRAUC.ipynb` notebook, yielding similar results.

Embedded methods (`SelectFromModel` functionality) for feature ranking in Random Forest (RF) Classifiers were used for feature selection to reduce feature space dimensionality. 
RF Classifiers are particularly appropriate for the NR-AR dataset because they are
- insensitive to **multicolinearity** in the data set, because selection of random subsets of features at each split; the effect of redundant features is diluted, allowing focus on the more informative ones
- robust to **high-dimensionality** due to inherent randomness in feature subset selection
- capable of handling **imbalanced datasets** by focussing on both classes during tree building when improving classification accuracy.
Feature selection was done in the `RF_feature_select.ipynb` notebook. The 20 most important features were selected, and a new RF Classifier was generated in the reduced 20-dimensional feature space. Performance metrics were very close to those from the full-dimensional RF model, indicating the selected features are sufficient for model building.

SMOTE (Synthetic Minority Over-sampling Technique) was tested to improve the performance of RF models 
in imbalanced datasets. SMOTE addresses the issues with modeling the minority class (toxic) by generating synthetic minority class samples. Instead of simply duplicating the minority class samples, SMOTE creates new, synthetic data points by interpolating between existing samples in the minority class. This balances the class distribution in the training set, allowing the Random Forest model to learn from a more representative dataset. SMOTE was imported from the `imblearn` package and used to resample the training set. Model building, hyperparameter tuning and cross-validation were performed similar as before in the `SMOTE_RandForrClass.ipynb` and `SMOTE_RandForrClass_pipeline,ipynb` notebooks. Performance metrics are similar to those obtained without SMOTE, with obtained precisions showing a mild decrease. Feature selection was repeated in the `SMOTE_RF_feature_select.ipynb` notebook.

## Prerequisite Software 

 - sklearn
 - rdkit
 - pickle
 - imblearn
 

## Data Files

- `NR-AR_processeddata.pkl`: pickle file containing the preprocessed training and test data as well as the corresponding toxicity labels. Smiles and molecular identification codes are also saved. Data generated in `data_preprocessing.ipynb`.
-  `tox_corrmat.csv`: csv-file containing correlation matrix of all features and toxicity label, ordered according to decreasing correlation to the toxicity label. Data generated in `data_preprocessing.ipynb`.
- `param_sel_rf1.npy`: numpy file containing list of 20 selected features from Random Forest Classifier embedded feature selection in `RF_feature_select.ipynb`.
- `SMOTE_param_sel_rf1.npy`: numpy file containing list of 20 selected features from Random Forest Classifier embedded feature selection using SMOTE resampled training data generated in `RF_feature_select.ipynb`.



