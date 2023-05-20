## To install all the required packages:
pip install -r requirements.txt

## Usage
To use this code, run the main() function in the script. This will load the input data, preprocess it, perform feature
selection, and train models for clustering and classification. 
The input data should be in a file called api.txt in the data folder.

## Functions
This code includes the following functions:

load_input_data(filename): loads and cleans the input data
preprocess_data(data): preprocesses and balances the training data
perform_feature_selection(data): performs feature selection on preprocessed data
start_clustering(data): performs clustering on the input data
main(): the driver code