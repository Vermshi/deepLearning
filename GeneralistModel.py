import torch.nn as nn
import torch

class GeneralistModel(nn.Module):
    def __init__(self, inputSize, nHidden):
        super(GeneralistModel, self).__init__()
        self.nHidden = nHidden
        self.inputSize = inputSize
        self.gru = nn.GRU(inputSize, nHidden,1)
        self.out = nn.Linear(nHidden, inputSize)
        self.outFunction = nn.Sigmoid()

    def forward(self, input, hidden):
        hidden =torch.add(hidden, torch.Tensor(1, 1, self.nHidden).uniform_(-5, 5))
        output, hidden = self.gru(input, hidden)
        if(self.outFunction == None):
            output = self.out(output[0])
        else:
            output = self.outFunction(self.out(output))


        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.nHidden)

