import torch.nn as nn
import torch

class GeneralistModel(nn.Module):
    def __init__(self, inputSize, nHidden):
        super(GeneralistModel, self).__init__()
        self.nHidden = nHidden

        self.gru = nn.GRU(inputSize, nHidden)
        self.out = nn.Linear(nHidden, inputSize)
        self.outFunction = nn.Softmax()

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        if(self.outFunction == None):
            output = self.out(output[0])
        else:
            output = self.outFunction(self.out(output[0]))


        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.nHidden)

