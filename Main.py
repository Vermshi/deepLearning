import matplotlib.pyplot as plt
from helpers.dataset import *
from Specialist import *
import torch
from helpers.datapreparation import *
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

    # Evaluation code--------------------------------------------------------
    binOutputs = []
    for i in range(inder.shape[0]):
        singleTarget = targets[i]
        singleOutput = outputs[i]
        # print("out")
        nTangents = getNumberOfTangents(singleTarget)
        binout = binarizeOutput(singleOutput,nTangents)
        binOutputs.append(binout)
        whatIsOutput = []
        for i in range(singleTarget.shape[1]):
            if(singleTarget[0][i] == 1):
                whatIsOutput.append(binout[i])

        # print("whatisoutputontargetindex")
        # print(whatIsOutput)
        totalStats[0]+=whatIsOutput.count(1)
        totalStats[1]+=nTangents

    #Vizulization code-------------------------------------------------------
    # binOutputs = np.array(binOutputs).transpose()
    # visualize_piano_roll(binOutputs,fs=fs)

    return totalStats

def getNumberOfTangents(singleTarg):
    numberOfTan = 0
    for i in range(singleTarg.shape[1]):
        if (singleTarg[0][i] == 1):
            numberOfTan+=1
    return numberOfTan

def composeMusic(inder,timeSteps):
    hidden = rnn.initHidden()
    randomTag = torch.tensor([[0]])
    binOutputs = []
    nTangents = 5
    for line in inder:
        output, hidden = rnn(line.view(1,1,line.shape[1]), randomTag, hidden)
        binout = binarizeOutput(output[0], nTangents)
        binOutputs.append(binout)

    binout = binarizeOutput(output[0], nTangents)
    binOutputs.append(binout)
    tensorBinout = torch.tensor(binout).view(1,1, -1).float()
    output = tensorBinout

    for i in range(timeSteps):
        # print("out")
        output, hidden = rnn(output, randomTag, hidden)

        binout = binarizeOutput(output[0], nTangents)
        binOutputs.append(binout)
        tensorBinout = torch.tensor(binout).view(1,1,-1).float()
        output = tensorBinout

    # Vizulization code-------------------------------------------------------
    binOutputs = np.array(binOutputs).transpose()
    visualize_piano_roll(binOutputs,fs=fs)

def getClosestIndex(numTags,hidden,nHidden):
    distances = []
    embed = nn.Embedding(numTags, nHidden)
    for i in range(numTags):
        #Gets eucluidian distance between the vectors
        distances.append(torch.dist(embed(torch.tensor([[i]])),hidden))
    closestIndex = distances.index(min(distances))
    return torch.tensor([[closestIndex]])




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
    temperature = 5
    dataobj = pianoroll_dataset_chunks("datasets/training/piano_roll_fs"+str(fs))
    dataobj.gen_batch()
    input, tag, target = dataobj.__getitem__(0)
    # ----------------------------------------------- Play code
    pianorollMatrix = input.view(input.shape[2],input.shape[0]).numpy()
    v = embed_play_v1(pianorollMatrix,fs=fs)
    # ---------------------------------- End
    numTags = dataobj.num_tags()
    inputSize = input.shape[2]
    nGRUS = input.shape[0]
    nHidden = 256
    learningRate = 0.01
    epochs = 100
    miniBatchSize = 3
    #Used because categorical cross entropy doesnt work
    lossFunction = torch.nn.BCELoss()
    rnn = Specialist(inputSize,nHidden,numTags,temperature)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learningRate)

    outputs = []
    errors = []

    for i in range(epochs):
        for k in range(miniBatchSize):
            print("Blasting bach through the system: ", (k + 1)+(i*miniBatchSize), "/", miniBatchSize*epochs)
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

    print("The network got", totalStats[0], " / ", totalStats[1], "tangents hit correctly.")
    print("The networked guessed the right composer", totalStats[2], " of ", totalStats[3], " times.")
    composeMusic(input[:5],50)
    gen_music(rnn,length=50,fs=fs)
