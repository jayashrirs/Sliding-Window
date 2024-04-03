#This file has all the declarations of the required classes and functions

#Importing Necessary Libraries
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


# Define the dataset class to load input attributes
class VoltageDataset(Dataset):
    def __init__(self, file_path):
        """
        Initialize the VoltageDataset with the file path.

        Parameters:
        - file_path (str): The path to the CSV file containing voltage data.
        """
        self.data = pd.read_csv(file_path)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx, :].values, dtype=torch.float32)
        return sample['time','voltage','temperature','vdd', 'pd', 'vinp']

#Class Declarations
class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, process_type):
        """
        Initialize the TransformerModel.

        Parameters:
        - num_layers (int): Number of layers in the transformer.
        - d_model (int): Dimensionality of the model.
        - num_heads (int): Number of attention heads.
        - dff (int): Dimensionality of the feed-forward layer.
        - input_vocab_size (int): Size of the input vocabulary.
        - target_vocab_size (int): Size of the target vocabulary.
        - pe_input (int): Maximum input sequence length for positional encoding.
        - pe_target (int): Maximum target sequence length for positional encoding.
        - process_type (str): Type of process.
        """
        super(TransformerModel, self).__init__()
        # Initialize the encoder
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, process_type)
        # Initialize the decoder
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, process_type)
        # Linear layer for final output
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        Forward pass of the TransformerModel.

        Parameters:
        - inp (tensor): Input tensor.
        - tar (tensor): Target tensor.
        - training (bool): Whether the model is in training mode.
        - enc_padding_mask (tensor): Mask for encoder padding.
        - look_ahead_mask (tensor): Mask for preventing future information in decoder.
        - dec_padding_mask (tensor): Mask for decoder padding.

        Returns:
        - final_output (tensor): Final output tensor.
        - attention_weights (dict): Attention weights.
        """
        # Encode the input
        enc_output = self.encoder(inp, training, enc_padding_mask)
        # Decode using encoder output
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        # Linear transformation for final output
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Initialize the EncoderLayer.

        Parameters:
        - d_model (int): Dimensionality of the model.
        - num_heads (int): Number of attention heads.
        - dff (int): Dimensionality of the feed-forward layer.
        - rate (float): Dropout rate.
        """
        super(EncoderLayer, self).__init__()
        # Multi-Head Attention layer
        self.mha = MultiHeadAttention(d_model, num_heads)
        # Feed-Forward Network
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        # Layer normalization
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        # Dropout layers
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, training, mask):
        """
        Forward pass of the EncoderLayer.

        Parameters:
        - x (tensor): Input tensor.
        - training (bool): Whether the model is in training mode.
        - mask (tensor): Mask for preventing attention to certain positions.

        Returns:
        - out2 (tensor): Output tensor.
        """
        # Multi-Head Attention
        attn_output, _ = self.mha(x, x, x, mask)
        # Apply dropout and add residual connection
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        # Feed-Forward Network
        ffn_output = self.ffn(out1)
        # Apply dropout and add residual connection
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, process_type):
        """
        Initialize the Encoder.

        Parameters:
        - num_layers (int): Number of layers in the encoder.
        - d_model (int): Dimensionality of the model.
        - num_heads (int): Number of attention heads.
        - dff (int): Dimensionality of the feed-forward layer.
        - input_vocab_size (int): Size of the input vocabulary.
        - maximum_position_encoding (int): Maximum position for positional encoding.
        - process_type (str): Type of process.
        """
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        # Embedding layer
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        # Positional encoding
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        # Encoder layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
        self.process_type = process_type

    def forward(self, x, training, mask):
        """
        Forward pass of the Encoder.

        Parameters:
        - x (tensor): Input tensor.
        - training (bool): Whether the model is in training mode.
        - mask (tensor): Mask for preventing attention to certain positions.

        Returns:
        - x (tensor): Output tensor.
        """
        seq_len = x.shape[1]
        # Embedding and scaling
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        # Positional encoding
        x += self.pos_encoding[:, :seq_len, :]
        # Permute dimensions for compatibility with EncoderLayer
        x = x.permute(1, 0, 2) # (batch_size, seq_len, d_model)
        # Pass through encoder layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Initialize the DecoderLayer.

        Parameters:
        - d_model (int): Dimensionality of the model.
        - num_heads (int): Number of attention heads.
        - dff (int): Dimensionality of the feed-forward layer.
        - rate (float): Dropout rate.
        """
        super(DecoderLayer, self).__init__()
        # Multi-Head Attention layers
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        # Feed-Forward Network
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        # Layer normalization
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        # Dropout layers
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass of the DecoderLayer.

        Parameters:
        - x (tensor): Input tensor.
        - enc_output (tensor): Encoder output tensor.
        - training (bool): Whether the model is in training mode.
        - look_ahead_mask (tensor): Mask for preventing future information.
        - padding_mask (tensor): Mask for preventing attention to padded tokens.

        Returns:
        - out3 (tensor): Output tensor.
        - attn_weights_block1 (tensor): Attention weights for the first attention block.
        - attn_weights_block2 (tensor): Attention weights for the second attention block.
        """
        # Multi-Head Attention 1
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        # Apply dropout and add residual connection
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        # Multi-Head Attention 2 with encoder output
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        # Apply dropout and add residual connection
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        # Feed-Forward Network
        ffn_output = self.ffn(out2)
        # Apply dropout and add residual connection
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, process_type):
        """
        Initialize the Decoder.

        Parameters:
        - num_layers (int): Number of layers in the decoder.
        - d_model (int): Dimensionality of the model.
        - num_heads (int): Number of attention heads.
        - dff (int): Dimensionality of the feed-forward layer.
        - target_vocab_size (int): Size of the target vocabulary.
        - maximum_position_encoding (int): Maximum position for positional encoding.
        - process_type (str): Type of process.
        """
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        # Embedding layer
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        # Positional encoding
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        # Decoder layers
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
        self.process_type = process_type

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass of the Decoder.

        Parameters:
        - x (tensor): Input tensor.
        - enc_output (tensor): Encoder output tensor.
        - training (bool): Whether the model is in training mode.
        - look_ahead_mask (tensor): Mask for preventing future information.
        - padding_mask (tensor): Mask for preventing attention to padded tokens.

        Returns:
        - x (tensor): Output tensor.
        - attention_weights (dict): Attention weights.
        """
        seq_len = x.shape[1]
        attention_weights = {}
        # Embedding and scaling
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        # Positional encoding
        x += self.pos_encoding[:, :seq_len, :]
        # Permute dimensions for compatibility with DecoderLayer
        x = x.permute(1, 0, 2) # (batch_size, seq_len, d_model)
        # Pass through decoder layers
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        return x, attention_weights
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Initialize the MultiHeadAttention layer.

        Parameters:
        - d_model (int): Dimensionality of the model.
        - num_heads (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        # Linear transformations for queries, keys, and values
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        # Final linear transformation
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).

        Parameters:
        - x (tensor): Input tensor.
        - batch_size (int): Batch size.

        Returns:
        - x (tensor): Tensor after splitting heads.
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        """
        Forward pass of the MultiHeadAttention layer.

        Parameters:
        - v (tensor): Value tensor.
        - k (tensor): Key tensor.
        - q (tensor): Query tensor.
        - mask (tensor): Mask for preventing attention to certain positions.

        Returns:
        - output (tensor): Output tensor.
        - attention_weights (tensor): Attention weights.
        """
        batch_size = q.shape[0]
        # Linear transformations
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # Scaled dot-product attention
        scaled_attention_logits = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        # Softmax
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        # Weighted sum
        output = torch.matmul(attention_weights, v)
        # Concatenate heads
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        # Final linear transformation
        return self.dense(output), attention_weights



#Functions
    

def point_wise_feed_forward_network(d_model, dff):
    """
    Create a point-wise feed-forward network.

    Parameters:
    - d_model (int): Dimensionality of the model.
    - dff (int): Dimensionality of the feed-forward layer.

    Returns:
    - network (Sequential): Sequential container for the feed-forward network.
    """
    # Define the feed-forward network using nn.Sequential
    network = nn.Sequential(
        nn.Linear(d_model, dff),  # Fully connected layer with input size d_model and output size dff
        nn.ReLU(),                # ReLU activation function
        nn.Linear(dff, d_model)   # Fully connected layer with input size dff and output size d_model
    )
    return network

def positional_encoding(position, d_model):
    """
    Generate positional encodings.

    Parameters:
    - position (int): Maximum sequence length.
    - d_model (int): Dimensionality of the model.

    Returns:
    - pos_encoding (tensor): Positional encodings tensor of shape (1, position, d_model).
    """
    # Generate angles using get_angles function
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # Apply sine to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cosine to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # Add batch dimension
    pos_encoding = angle_rads[np.newaxis, ...]
    # Convert to tensor
    return torch.tensor(pos_encoding, dtype=torch.float32)

def get_angles(pos, i, d_model):
    """
    Calculate the angles for positional encoding.

    Parameters:
    - pos (ndarray): Array representing positions in the sequence.
    - i (ndarray): Array representing dimensions of the model.
    - d_model (int): Dimensionality of the model.

    Returns:
    - angle_rates (ndarray): Array containing the angle rates for positional encoding.
    """
    # Calculate the angle rates using a formula that depends on the position and dimensionality of the model
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    # Multiply the position by the angle rates
    return pos * angle_rates

def evaluate(y_true, y_pred):
    """
    Calculate evaluation metrics for predicted values.

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - rmse (float): Root Mean Squared Error.
    - r2 (float): R-squared score.
    - mae (float): Mean Absolute Error.
    - snr (float): Signal-to-Noise Ratio.
    """
    # Root Mean Squared Error
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # R-squared score
    r2 = r2_score(y_true, y_pred)
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    # Signal-to-Noise Ratio
    snr = signal_to_noise_ratio(y_true, y_pred)  # You need to implement this function

    return rmse, r2, mae, snr