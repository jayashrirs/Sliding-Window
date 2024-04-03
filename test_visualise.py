import pickle
import os

# Load the function from the pickle file
with open('my_predict.pickle', 'rb') as f:
    loaded_test_function = pickle.load(f)

# Test the reloaded function
files=os.listdir("test")[1:]
print(len(files))
print(files)
for i in files:
    loaded_test_function(i)