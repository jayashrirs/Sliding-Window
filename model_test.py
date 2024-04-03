#This file uses the trained models to predict the waveform and visualise.

#Importing Necessary Libraries
from declarations import *

import os
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import signal_to_noise_ratio

from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define training and testing data directories
train_dir = "train"
test_dir = "test"

process_types = ['typical', 'fastnfastp', 'slownslowp', 'fastnslowp', 'slownfastp']



# Testing loop with plotting
def test_file(files):
  
  """
    Evaluate the Transformer model on the test dataset.

    Parameters:
    - model (nn.Module): Transformer model.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - criterion (callable): Loss function (not used here but kept for consistency).
    - device (torch.device): Device to perform computations (e.g., 'cuda' or 'cpu').

    Returns:
    - avg_rmse (float): Average Root Mean Squared Error.
    - avg_r2 (float): Average R-squared score.
    - avg_mae (float): Average Mean Absolute Error.
    - avg_snr (float): Average Signal-to-Noise Ratio.
    """

  test_metrics = {}
  for process_type in process_types:

      test_files = os.listdir(os.path.join(test_dir, process_type)) # fetch all test files
      process_metrics = {'RMSE': [], 'R2 Score': [], 'MAE': [], 'SNR': []} # initialise a dictionary for metrics

      #iterate each test file
      for test_file in test_files:
        #input 0-100 and predict 100-120, input 20-120 predict 120-140, etc
        for i in range(100,14880):
          #pick the model
          model="models/"+p+"_"+str(i)+"_to_"+str(i+100)+".h5"
          #load dataset
          test_dataset = VoltageDataset(os.path.join(test_dir, process_type, test_file))[i:i+100]
          test_loader = DataLoader(test_dataset, batch_size=batch_size)
          original_values = []
          predicted_values = []
          #predict using models
          with torch.no_grad():
            for inputs, targets in test_loader:
                # Move inputs and targets to the specified device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass without gradient tracking
                predictions, _ = model(inputs.unsqueeze(0), targets.unsqueeze(0), False, None, None, None)
                predictions = predictions.squeeze(0)

                # Convert predictions and targets to CPU and numpy arrays, and extend lists
                original_values.extend(targets.cpu().numpy())
                predicted_values.extend(predictions.cpu().numpy())


          #metrics
          rmse, r2, mae, snr = evaluate(original_values, predicted_values)
          print("File: ",test_file)
          process_metrics['RMSE'].append(rmse)
          process_metrics['R2 Score'].append(r2)
          process_metrics['MAE'].append(mae)
          process_metrics['SNR'].append(snr)

          # Plotting
          plt.figure(figsize=(10, 6))
          plt.plot(original_values[:20], label='Original')
          plt.plot(predicted_values[:20], label='Predicted')
          plt.title(f"Comparison of Original and Predicted Values for {test_file}")
          plt.xlabel("Time")
          plt.ylabel("Voltage")
          plt.legend()
        #   plt.savefig("plots/"+test_file[:-4]+".jpg") #saved once
          plt.show()

      # Calculate average metrics for the process type
      for metric, values in process_metrics.items():
          process_metrics[metric] = np.mean(values)

      test_metrics[process_type] = process_metrics

  # Display test metrics
  for process_type, metrics in test_metrics.items():
      print(f"Process Type: {process_type}")
      for metric, value in metrics.items():
          print(f"{metric}: {value}")

# files = os.listdir(os.path.join(test_dir))
# test_model(files)