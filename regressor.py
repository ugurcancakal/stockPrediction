'''
Stock Regressors
author: ugurc
201201
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
  def __init__(self, n_features, seq_length,
               hidden_size=50, n_layers=4, dropout=.2, output_size=1):
    '''
    Args:
      n_features : int
        number of different features, (parallel inputs). In the stock prize 
        case, features are the indicators like openning and closing prices

      seq_length : int
        sequence length, number of timesteps
      
      hidden_size : int (default=50) 
        number of hidden states(features). In keras, it's units
      
      n_layers : int (default=4)
        Number of recurrent layers. E.g., setting ``n_layers=2``
        would mean stacking two LSTMs together to form a `stacked LSTM`,
        with the second LSTM taking in outputs of the first LSTM and
        computing the final results.
    
      dropout: int (default=.2)
        dropout probability of the dropout layer which is to be embedded right
        after each LSTM layer rather than the output layer

      output_size : (default = 1)
        number of output features, (parallel outputs). In the stock prize 
        case, features are the indicators like openning and closing prices

      Notes:
        Since batch_first = True, input and output tensors are provided as 
        (batch, seq, feature)
    '''
    
    super().__init__()
    self.n_features = n_features
    self.seq_length = seq_length
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.dropout = dropout
    self.output_size = output_size

    # Layers
    self.lstm = nn.LSTM(n_features, hidden_size, n_layers, 
                        dropout = dropout, batch_first = True)
    self.linear = nn.Linear(hidden_size,output_size)

  def forward(self, input_seq):
    # Set initial hidden and cell states
    # print(input_seq.shape)
    # input_seq = input_seq.t()
    batch_size = input_seq.size(0)
    hidden_cell = self.init_hidden(batch_size)
    output, hidden_cell = self.lstm(input_seq, hidden_cell)
    # output = output.reshape(-1,self.hidden_size)
    # print(output.shape)
    predictions = self.linear(output)
    # print(predictions.shape)
    return predictions[:,-1,:]

  def init_hidden(self, batch_size):
    h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
    c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
    # hidden = torch.zeros(self.n_layers * self.n_directions,
    #                      batch_size, self.hidden_size)
    return create_variable(h0), create_variable(c0)


def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)