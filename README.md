# ProteinSequenceClassifier
In this repo I'm adapting model from https://github.com/vkola-lab/peds2019 to classify protein sequences as either human or mouse proteins.

Tested on python version: 3.9.9

### Data generation script
Data for model in this repo is already generated, but if for some reason directory `data/sample/my_data` does not contain train, val, test split then one can use `data/sample/my_split.py` script to generate these files.

### Model
To see how it works open `training_and_evaluation.ipynb` and follow the code. By default fitting is commented out and my last model and training data is loaded in the following cell.

