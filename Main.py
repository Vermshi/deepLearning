import matplotlib.pyplot as plt
from helpers.dataset import *
import torch.nn as nn
from GeneralistModel import *
from Specialist import *
import torch
from torch.autograd import Variable

def evaluate(inder, tag, targets):
    hidden = rnn.initHidden()
    #[Correct tangents, total tangents, correct Artist, Total artists]
    totalStats = [0,0,0,0]
    outputs, hidden = rnn(inder,tag, hidden)
    #Embedding code ---------------------------------------------------
    guessedArtist = getClosestIndex(rnn.numTags,hidden,nHidden)
    totalStats[3]+=1
    if(guessedArtist == tag):
        totalStats[2]+=1

    #Embedding end-------------------------------------------------------
    for i in range(inder.shape[0]):
        singleTarget = targets[i]
        singleOutput = outputs[i]
        # print("out")
        nTangents = getNumberOfTangents(singleTarget)
        binout = binarizeOutput(singleOutput,nTangents)
        whatIsOutput = []
        for i in range(singleTarget.shape[1]):
            if(singleTarget[0][i] == 1):
                whatIsOutput.append(binout[i])

        # print("whatisoutputontargetindex")
        # print(whatIsOutput)
        totalStats[0]+=whatIsOutput.count(1)
        totalStats[1]+=nTangents


    return totalStats

def getNumberOfTangents(singleTarg):
    numberOfTan = 0
    for i in range(singleTarg.shape[1]):
        if (singleTarg[0][i] == 1):
            numberOfTan+=1
    return numberOfTan

def getClosestIndex(numTags,hidden,nHidden):
    distances = []
    embed = nn.Embedding(numTags, nHidden)
    for i in range(numTags):
        #Gets eucluidian distance between the vectors
        distances.append(torch.dist(embed(torch.tensor([[i]])),hidden))
    closestIndex = distances.index(min(distances))
    return torch.tensor([[closestIndex]])

# def train(target, input):
#     hidden = rnn.initHidden()
#
#     rnn.zero_grad()
#     outputs = []
#     errors = []
#     for i in range(input.shape[0]):
#         singleInput = input[i].view(1, 1, -1)
#         output, hidden = rnn(singleInput, hidden)
#         singleTarget = target[i].view( 1, -1)
#
#         loss = lossFunction(output, singleTarget)
#         loss.backward(retain_graph=True)
#
#         # Add parameters' gradients to their values, multiplied by learning rate
#         for p in rnn.parameters():
#             p.data.add_(-learningRate, p.grad.data)
#         errors.append(loss.item())
#         outputs.append(output)
#
#     return outputs, np.sum(errors)/len(errors)


def binarizeOutput(output,nTangs):
    normOut = output.detach().numpy()
    normOut = normOut
    sortedIndexes = np.argsort(-normOut)
    outputBinary = [0]*output.shape[1]
    for i in range(nTangs):
        outputBinary[sortedIndexes[0][i]] = 1


    return outputBinary


if __name__ == '__main__':
    # Get a dataset by sending in index to getitem:
    dataobj = pianoroll_dataset_chunks("datasets/training/piano_roll_fs5")
    dataobj.gen_batch()
    input, tag, target = dataobj.__getitem__(0)
    numTags = dataobj.num_tags()
    inputSize = input.shape[2]
    nGRUS = input.shape[0]
    nHidden = 256
    learningRate = 0.01
    epochs = 1
    miniBatchSize = 100

    lossFunction = torch.nn.BCELoss()
    rnn = Specialist(inputSize,nHidden,numTags)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learningRate)

    outputs = []
    errors = []

    for i in range(epochs):
        for k in range(miniBatchSize):
            print("Blasting bach through the system: ", (k + 1)*(i+1), "/", miniBatchSize*epochs)
            hidden = None
            optimizer.zero_grad()
            input, tag, target = dataobj.__getitem__(k)
            output, hidden = rnn(input, tag,hidden=hidden)
            loss = lossFunction(output, target)
            loss.backward(retain_graph=False)
            optimizer.step()
            errors.append(loss.item())

    plt.figure()
    plt.plot(errors)
    plt.show()
    totalStats = [0,0,0,0]
    for i in range(miniBatchSize):
        print("Evaluating song: ", i+1, "/", miniBatchSize)
        with torch.no_grad():
            input, tag, target = dataobj.__getitem__(i)
            stats = evaluate(input,tag,target)
            totalStats[0]+=stats[0]
            totalStats[1]+=stats[1]
            totalStats[2]+=stats[2]
            totalStats[3]+=stats[3]

    print("The network got", totalStats[0], " / ", totalStats[1], "correct.")
    print("The networked guessed the right composer", totalStats[2], " of ", totalStats[3], " times.")