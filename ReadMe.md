# ML
Project for the "machine learning" course for the master in computer science

## Getting started

Let's first take a look at the project organization and its requirements.

### Structure
The project is structured in the following way:
* DATA = folder containing dataset for training , test and validation for the Cup and Monk. There are also the results in CSV format of various scripts.
* Script = folder with the python scripts of tests and computing, for each model.
* Image = folder containing the plots and the heatmap generated from the python's script.

### Prerequisites
The entire project was made with python 3.6+ and the following libraries are necessary for the operation:

* numpy
* matplotlib
* seaborn
* pandas
* scikit-learn
* keras
* tensorflow

## Running the tests

Before seeing what kind of tests we can do, let's see which other component we can found in the folders.

### Scripts
In the folder we can find the following scripts:
* Script/MLP/NNGridSearch.py = script for the first grid search about a Multi Layer Perception.
* Script/MLP/MPL_newValidation.py = script for the optimization of the MLP.
* Script/MONK/Monk.py = the script implements the grid search and the evaluation of A MLP for the Monk problems.
* Script/RF/ETR.py = it implements the grid search and evaluation of a Extremely Random Tree.
* Script/RF/RFR.py = it implements the grid search and evaluation of a Random Forest Regressor.
* Script/SVR/SVM.py = it implements the first grid search for two kind of Support Vector Regressor: Polynomial and RBF.
* Script/SVR/SVM_newGridS.py = it implements the second grid search for the optimization for both type of SVR. 
* Script/Test.py = script for assessment and predict
* Script/utils.py = a script containing all support functions.
* Script/config.py = a support script that stores global variables like Path and Score function.
* Script/split_data.py = a support script to generate the training and test set. If argument is 'new', it will generate a new training and validation set from the first training set.

### MLP
To make the grid search to study the behaviors on different architectures, we need to launch the script:
```
python3 NNGridSearch.py -> DATA/NN/grid_search_result_MLP.csv
```
This will produce a CSV file with the table and it will be necessary for the second script, that will take the best result and it will plot the learning curve:
```
python3 MPL_newValidation.py -> Image/NN/Eta0.001batch64m0.9epochs5000.png
```
The learning curve is made from a different training and validation set (to do a "sub-model selection"). The parameter change is manual. If you want to make a CrossValidation on the model selected, type the following:
```
python3 MPL_newValidation.py cv
```

### SVM
The two different type of SVR are performed together on Gamma,degree and C (with orders of different sizes). We need to launch the script:
```
python3 SVM.py -> DATA/SVR_POLY/grid_search_result_SVR_POLY.csv and DATA/SVR_RBF/grid_search_result_SVR_RBF.csv
               -> Image/SVR_RBF/epsilon_0.01_with_meeSVR_RBF10_56.png and Image/SVR_POLY/epsilon_0.01_with_meeSVR_POLY16_56.png
```
To run the second script we need the csv, for both models, with the result of the first one. The following script
```
python3 SVM_newGridS.py -> DATA/SVR_POLY/grid_search_result_SVR_POLY2.csv and DATA/SVR_RBF/grid_search_result_SVR_RBF2.csv
                        -> Image/SVR_RBF/epsilon_0.01_with_meeSVR_RBF10_56.png and Image/SVR_POLY/epsilon_0.01_with_meeSVR_POLY16_56.png
```
 will produce new results by taking the best two different architecture and generate an interval of values where run a new grid search.
 
### RFR
For these two models there are only two scripts, one for each one, and each of them need an argument to run:

```
python3 RFR.py -i grid      -> DATA/RFR/grid_search_result_RFR.csv
python3 ETR.py -i grid      -> DATA/ETR/grid_search_result_ETR.csv
python3 RFR.py -i cv        -> evaluation
python3 ETR.py -i cv        -> evaluation
python3 RFR.py -i predict   -> DATA/Results/blind_RFR.csv
python3 ETR.py -i predict   -> DATA/Results/blind_ETR.csv
```
If the given argument is 'grid', it will be run a grid search and save the results on a CSV file.
If the given argument is 'cv' and the CSV file with results exists, the best model it will be evaluated with a k-fold cross validation.
If the given argument is 'predict', the best model will train on the full dataset and it will predict the solution for the CUP.

### Test and Blind
At least, for evaluate each model it's necessary to run the following script:

```
python3 Test.py 
```
It will produce the MEE error for the best NN model, the best SVM model and the best RF model. Instead, if you want to 
predict the data for the ML CUP with the model that won the model selection, just add the argument 'predict' as follow:
```
python3 Test.py predict
```