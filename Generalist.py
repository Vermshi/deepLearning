import matplotlib.pyplot as plt
from helpers.dataset import *
import torch.nn as nn
from GeneralistModel import *
import torch

def train(target, input):
    hidden = rnn.initHidden()

    rnn.zero_grad()
    outputs = []
    errors = []
    for i in range(input.shape[0]):
        singleInput = input[i].view(1, 1, -1)
        output, hidden = rnn(singleInput, hidden)
        singleTarget = target[i].view( 1, -1)
        print("out")
        print(binarizeOutput(output))
        print(output)
        print("targ")
        print(singleTarget)
        loss = lossFunction(output, singleTarget)
        loss.backward(retain_graph=True)

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
            p.data.add_(-learningRate, p.grad.data)
        errors.append(loss.item())
        outputs.append(outputs)

    return outputs, np.sum(errors)/len(errors)


def binarizeOutput(output):

    tresh = 0.01

    t = torch.Tensor([tresh])  # threshold
    output = (output > t).float() * 1

    return output


if __name__ == '__main__':
    # Get a dataset by sending in index to getitem:
    dataobj = pianoroll_dataset_chunks("datasets/training/piano_roll_fs5")
    dataobj.gen_batch()
    input, tag, target = dataobj.__getitem__(0)
    print(tag)
    print(input.shape)
    print(tag.shape)
    print(target.shape)
    inputSize = input.shape[2]
    nGRUS = input.shape[0]
    nHidden = 130
    learningRate = 0.05
    epochs = 10
    miniBatchSize = 10

    lossFunction = torch.nn.MSELoss()
    # Takes in input_size, hidden
    rnn = GeneralistModel(inputSize, nHidden)
    # initHidden = torch.zeros(1, 1, nHidden)
    # singleInput = input[0].view(1, 1, -1)
    # output, nextHidden = rnn(singleInput, initHidden)
    allErrors = []
    for i in range(epochs):
        for i in range(miniBatchSize):
            input, tag, target = dataobj.__getitem__(i)
            _,caseError = train(target,input)
            allErrors.append(caseError)

    plt.figure()
    plt.plot(allErrors)
    plt.show()