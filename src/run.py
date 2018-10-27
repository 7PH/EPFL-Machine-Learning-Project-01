from src.helpers import load_csv_data

# Constants
DATA_FOLDER = '../data/'

# Load data
y_train, x_train, x_ids = load_csv_data(DATA_FOLDER + "train.csv")
y_test, x_test, x_test_ids = load_csv_data(DATA_FOLDER + "test.csv")

# Do stuff
# @TODO
