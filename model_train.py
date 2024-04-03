#This file trains the models and saves them so that thay can be used for prediction

#Importing Necessary Libraries
from declarations import *

import os
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_log_error
#from sklearn.metrics import signal_to_noise_ratio

from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



def train(model, train_loader, criterion, optimizer, device):
    """
    Train the Transformer model.

    Parameters:
    - model (nn.Module): Transformer model.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - criterion (callable): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - device (torch.device): Device to perform computations (e.g., 'cuda' or 'cpu').

    Returns:
    - total_loss (float): Average training loss.
    """
    # Set the model to training mode
    model.train()
    total_loss = 0
    # Iterate over batches in the training data
    for batch in train_loader:
        # Move inputs and targets to the specified device
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        predictions, _ = model(inputs.unsqueeze(0), targets.unsqueeze(0), True, None, None, None)
        # Compute the loss
        loss = criterion(predictions.squeeze(0), targets)
        # Backpropagation
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Accumulate the total loss
        total_loss += loss.item()
    # Calculate the average training loss
    return total_loss / len(train_loader)



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
input_vocab_size = 10000
target_vocab_size = 10000
pe_input = 1000
pe_target = 1000
process_types = ['typical', 'fastnfastp', 'slownslowp', 'fastnslowp', 'slownfastp']
num_epochs = 100
batch_size = 32
learning_rate = 0.001
batch_size

# Create model instances for each process type
models = {}  # Create an empty dictionary to store models
for process_type in process_types:
    # Create a Transformer model for the current process type and add it to the models dictionary
    models[process_type] = TransformerModel(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, process_type).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Define Mean Squared Error loss as the criterion

optimizers = {}  # Create an empty dictionary to store optimizers

for process_type in process_types:
    # Create an Adam optimizer for the model parameters of the current process type
    optimizers[process_type] = torch.optim.Adam(models[process_type].parameters(), lr=learning_rate)

# Define training and testing data directories
train_dir = "train"
test_dir = "test"

# Training loop
for epoch in range(num_epochs):
    for process_type in process_types:
        model = models[process_type]
        optimizer = optimizers[process_type]

        # Load training data for the current process type
        train_files = os.listdir(os.path.join(train_dir, process_type))
        train_losses = []

        # Iterate over training files
        for train_file in train_files:
            # Load the entire training dataset
            #input 0-100 and predict 100-120, input 20-120 predict 120-140, etc
            for i in range(0,14880,100):
              train_dataset = VoltageDataset(os.path.join(train_dir, process_type, train_file))[i:i+100]
              train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

              # Training for each batch
              for inputs, targets in train_loader:
                  inputs, targets = inputs.to(device), targets.to(device)

                  # Zero the gradients
                  optimizer.zero_grad()

                  # Forward pass and compute loss
                  predictions, _ = model(inputs.unsqueeze(0), targets.unsqueeze(0), True, None, None, None)
                  loss = criterion(predictions.squeeze(0), targets)

                  # Backward pass and optimization step
                  loss.backward()
                  optimizer.step()

                  # Append the loss to the list of losses for this epoch
                  train_losses.append(loss.item())

            # Save the model after each epoch for the current process type
            model_save_path = "models/"+process_type+"_"+str(i)+"_to_"+str(i+100)+".h5"
            torch.save(model.state_dict(), model_save_path)

