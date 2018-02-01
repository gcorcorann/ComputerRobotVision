import torch
import torch.nn as nn
from torch.autograd import Variable

# set random seed
torch.manual_seed(1)

inputs = [Variable(torch.randn(1, 3)) for _ in range(5)]
print(inputs)
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
print(inputs)

## hyper-parameters
#batch_size = 5
#lstm_hidden = 4
#seq_length = 10
#input_size = (100, 100)
#
## create inputs
#inputs = torch.randn(seq_length, batch_size, 3, *input_size)
## reshape inpot [numSeqs x batchSize x numFeats]
#inputs = inputs.view(seq_length, batch_size, -1)
## store in Variables
#inputs = Variable(inputs)
#print('inputs:', inputs.size())
#
## create LSTM
#lstm = nn.LSTM(input_size[0]*input_size[1]*3, lstm_hidden, 1)
#
## initialize hidden and cell layer
#hidden = (Variable(torch.zeros(1, batch_size, lstm_hidden)),
#        Variable(torch.zeros(1, batch_size, lstm_hidden)))
#
## for each input in sequence
#for i, inp in enumerate(inputs):
#    print(i, 'inp:', inp.size())
#    # reformat into [1 x batchSize x numFeats]
#    inp = inp.unsqueeze(0)
#    # pass through lstm
#    outputs, hidden = lstm.forward(inp, hidden)
#    print('outputs:', outputs.size())
#
#print(outputs[-1])
#
## initialize hidden and cell layer
#hidden = (Variable(torch.zeros(1, batch_size, lstm_hidden)),
#        Variable(torch.zeros(1, batch_size, lstm_hidden)))
#
#outputs, hidden = lstm(inputs)
#print(outputs.size())
#print(outputs[-1])
