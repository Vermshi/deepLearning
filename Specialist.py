import torch
import torch.nn as nn

class Specialist (nn.Module):
    def __init__ (self, inputSize , hiddenSize , numTags ,temperature, nLayers=1):
        super(Specialist , self). __init__()
        self. inputSize = inputSize
        self. hiddenSize = hiddenSize
        self. numTags = numTags
        self. nLayers = nLayers
        self.temperature = temperature

        self.tagsEmbedding = nn.Embedding(numTags, hiddenSize)
        self. inputLayer = nn.Linear(in_features = inputSize, out_features = hiddenSize)
        self.gru = nn.GRU( hiddenSize , hiddenSize , nLayers )
        self. outputLayer = nn.Linear(in_features = hiddenSize, out_features = inputSize)
        self.output= nn.Sigmoid()

    def forward(self, inputSequence , tag , hidden=None):
        if(hidden is None):
            hidden = self.tagsEmbedding(tag)
        hidden = torch.add(hidden,torch.Tensor(1, 1,self.hiddenSize).uniform_(-self.temperature, self.temperature))
        output , hidden = self.gru(self.inputLayer(inputSequence), hidden)
        output = self.output(self. outputLayer (output))
        return output ,hidden

    def initHidden(self):
        return torch.zeros(self.nLayers, 1, self.hiddenSize)