import pickle
from test_function import test_file

# Save the function as a pickle file
with open('my_predict.pickle', 'wb') as f:
    pickle.dump(test_file, f)