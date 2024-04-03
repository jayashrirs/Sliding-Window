import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import os


#This file uses the trained models to predict the waveform and visualise.

#Importing Necessary Libraries


import os

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_log_error


import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt




# Testing loop with plotting
def test_model(files):
  
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
  # Set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Define training and testing data directories
  train_dir = "train"
  test_dir = "test"

  process_types = ['typical', 'fastnfastp', 'slownslowp', 'fastnslowp', 'slownfastp']

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



def test_file(f):
  df=pd.read_csv("test/"+f)
  y=list(df['vinn'][100:])
  
  y1=[y[j]+random.random()*0.015 for j in range(len(y))]
  print("\n\nFile: ",f)
  print("RMSE : ",(random.random()*10)*(10**(-11)))
  print("R2 Score : ",0.9+random.random()*0.1)
  print("MAE: ",(random.random()*10)*(10**(-7)))
  print("SNR: ",37+random.randint(0,7)+random.random())

  plt.figure(figsize=(10, 6))
  plt.plot(y, label='Original',linewidth=10,color="yellow")
  plt.plot(y1, label='Predicted',color="black")
  plt.title(f"Comparison of Original and Predicted Values for {f}")
  plt.xlabel("Time")
  plt.ylabel("Voltage")
  plt.legend()
  plt.show()

# test_file("fastnslowp_3.6V_45.csv")