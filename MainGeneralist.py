import matplotlib.pyplot as plt
from helpers.dataset import *
from GeneralistModel import *
import torch
from helpers.datapreparation import *
from torch.autograd import Variable


def evaluate(inder, targets):
    hidden = rnn.initHidden()
    #[Correct tangents, total tangents, correct Artist, Total artists]
    totalStats = [0,0]
    outputs, hidden = rnn(inder,hidden)

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
    fs = 1
    dataobj = pianoroll_dataset_batch("datasets/training/piano_roll_fs"+str(fs))
    dataobj.gen_batch()
    input, _, target = dataobj.__getitem__(0)
    # ----------------------------------------------- Viz code
    # print(input.shape)
    # pianorollMatrix = input.view(input.shape[2],input.shape[0]).numpy()
    # visualize_piano_roll(pianorollMatrix,fs=1)
    # v = embed_play_v1(pianorollMatrix,fs=1)
    # v()
    # exit()
    # ---------------------------------- End
    numTags = dataobj.num_tags()
    inputSize = input.shape[2]
    nGRUS = input.shape[0]
    nHidden = 256
    learningRate = 0.01
    epochs = 100
    miniBatchSize = 1
    #Used because categorical cross entropy doesnt work
    lossFunction = torch.nn.BCELoss()
    rnn = GeneralistModel(inputSize,nHidden)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learningRate)

    outputs = []
    errors = []

    for i in range(epochs):
        for k in range(miniBatchSize):
            print("Blasting bach through the system: ", (k + 1)+(i*miniBatchSize), "/", miniBatchSize*epochs)
            hidden = rnn.initHidden()
            optimizer.zero_grad()
            input, tag, target = dataobj.__getitem__(k)
            output, hidden = rnn(input, hidden)
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
            stats = evaluate(input,target)
            totalStats[0]+=stats[0]
            totalStats[1]+=stats[1]

    print("The network got", totalStats[0], " / ", totalStats[1], "tangents hit correctly.")
    gen_music(rnn, length=50, fs=fs,generalist=True)