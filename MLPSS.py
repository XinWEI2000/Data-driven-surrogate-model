import os
import torch
import numpy as np
import random
import time
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# create the dataset for the training and testing
class MyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        # 根据索引获取单个样本
        sampledata = self.data[index]
        sampletarget = self.target[index]
        return sampledata, sampletarget

# the function for reading the data from the files to create the training dataset and testing dataset
def readfile(xunlian, ceshi):
    # create the training data----------------------------------------------
    x = []
    z = []
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    Xdeformation = []
    layout = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 2, 3, 4, 3, 2, 2, 3, 2, 4, 3, 4, 3, 4, 2, 3, 1, 4, 2, 2],
              [1, 2, 2, 2, 2, 1, 1, 1, 4, 1, 3, 4, 3, 3, 2, 1, 1, 2, 1, 3],
              [4, 4, 3, 3, 2, 2, 2, 4, 3, 2, 1, 1, 4, 4, 4, 1, 3, 2, 1, 4],
              [3, 1, 3, 1, 2, 4, 1, 2, 2, 1, 4, 4, 4, 3, 3, 3, 2, 3, 2, 2],
              [1, 2, 3, 1, 1, 3, 3, 2, 1, 4, 3, 3, 4, 4, 4, 4, 1, 1, 4, 3],
              [3, 2, 1, 1, 4, 1, 1, 3, 4, 2, 3, 2, 2, 3, 4, 1, 2, 3, 3, 4],
              [3, 2, 2, 3, 3, 4, 3, 3, 3, 1, 4, 1, 3, 2, 1, 4, 2, 4, 2, 3],
              [4, 4, 3, 4, 1, 2, 1, 3, 3, 1, 3, 3, 1, 1, 2, 1, 2, 1, 1, 3],
              [1, 3, 3, 2, 4, 2, 2, 3, 4, 2, 2, 2, 4, 3, 3, 3, 1, 3, 1, 1],
              [4, 1, 4, 4, 3, 4, 2, 1, 3, 2, 1, 1, 3, 3, 4, 3, 1, 2, 1, 2],
              [4, 4, 1, 3, 2, 2, 2, 2, 1, 4, 4, 4, 1, 1, 3, 1, 4, 1, 4, 3],
              [1, 2, 3, 2, 4, 3, 1, 1, 1, 3, 2, 1, 1, 3, 4, 2, 2, 1, 4, 4],
              [2, 4, 3, 2, 1, 3, 1, 1, 3, 3, 4, 4, 4, 3, 4, 1, 3, 4, 3, 1],
              [3, 3, 3, 4, 2, 2, 2, 2, 4, 2, 4, 4, 4, 4, 1, 3, 4, 3, 1, 2],
              [4, 3, 3, 2, 3, 4, 4, 1, 4, 2, 1, 4, 1, 2, 1, 1, 4, 4, 4, 2],
              [4, 4, 2, 2, 1, 4, 2, 1, 2, 4, 3, 3, 4, 4, 2, 3, 4, 3, 2, 2],
              [4, 2, 4, 2, 4, 3, 1, 4, 1, 2, 3, 4, 2, 1, 3, 3, 2, 2, 3, 2],
              [3, 3, 3, 2, 1, 4, 4, 4, 4, 4, 1, 1, 3, 3, 4, 3, 2, 1, 1, 4],
              [1, 3, 2, 3, 2, 2, 4, 3, 1, 1, 4, 1, 1, 3, 3, 1, 2, 4, 4, 1]]
    for i in range(len(xunlian)):
        a = xunlian[i]
        objFilePath = "D:\\Filefolder\\UWM\\Second_seminar\\Independent_study\\pythonProject\\SS_for_AM\\Xdeformation\\G" + a + "_Xdeformation.txt"
        with open(objFilePath) as file:

            while 1:

                line = file.readline()
                if not line:
                    break
                strs = line.split("\t")
                if strs[0] == "Node Number":
                    continue
                else:
                    # print('strs:',strs)
                    if strs[2] == "2.5e-002":
                        x.append((float(strs[1])))
                        z.append((float(strs[3])))
                        a1.append((float(layout[i][0])))
                        a2.append((float(layout[i][1])))
                        a3.append((float(layout[i][2])))
                        a4.append((float(layout[i][3])))
                        a5.append((float(layout[i][4])))
                        a6.append((float(layout[i][5])))
                        a7.append((float(layout[i][6])))
                        a8.append((float(layout[i][7])))
                        a9.append((float(layout[i][8])))
                        a10.append((float(layout[i][9])))
                        a11.append((float(layout[i][10])))
                        a12.append((float(layout[i][11])))
                        a13.append((float(layout[i][12])))
                        a14.append((float(layout[i][13])))
                        a15.append((float(layout[i][14])))
                        a16.append((float(layout[i][15])))
                        a17.append((float(layout[i][16])))
                        a18.append((float(layout[i][17])))
                        a19.append((float(layout[i][18])))
                        a20.append((float(layout[i][19])))
                        Xdeformation.append((float(strs[4])))
    a = np.arange(0, len(x))
    random.shuffle(a)
    b = a[0:3600]
    x = np.array(x)
    x1 = x[b]
    z = np.array(z)
    z1 = z[b]
    a1 = np.array(a1)
    a1a = a1[b]
    a2 = np.array(a2)
    a2a = a2[b]
    a3 = np.array(a3)
    a3a = a3[b]
    a4 = np.array(a4)
    a4a = a4[b]
    a5 = np.array(a5)
    a5a = a5[b]
    a6 = np.array(a6)
    a6a = a6[b]
    a7 = np.array(a7)
    a7a = a7[b]
    a8 = np.array(a8)
    a8a = a8[b]
    a9 = np.array(a9)
    a9a = a9[b]
    a10 = np.array(a10)
    a10a = a10[b]
    a11 = np.array(a11)
    a11a = a11[b]
    a12 = np.array(a12)
    a12a = a12[b]
    a13 = np.array(a13)
    a13a = a13[b]
    a14 = np.array(a14)
    a14a = a14[b]
    a15 = np.array(a15)
    a15a = a15[b]
    a16 = np.array(a16)
    a16a = a16[b]
    a17 = np.array(a17)
    a17a = a17[b]
    a18 = np.array(a18)
    a18a = a18[b]
    a19 = np.array(a19)
    a19a = a19[b]
    a20 = np.array(a20)
    a20a = a20[b]
    Xdeformation = np.array(Xdeformation)
    Xdeformationa = Xdeformation[b]

    # create the test data-------------------------------------------
    xte = []
    zte = []
    a1te, a2te, a3te, a4te, a5te, a6te, a7te, a8te, a9te, a10te, a11te, a12te, a13te, a14te, a15te, a16te, a17te, \
    a18te, a19te, a20te = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    Xdeformationte = []
    layoutte = [[1, 3, 1, 4, 2, 2, 1, 2, 4, 2, 4, 4, 2, 2, 3, 1, 3, 2, 1, 2],
                [4, 2, 2, 1, 4, 4, 3, 2, 3, 4, 3, 3, 4, 4, 3, 4, 4, 3, 3, 1],
                [2, 1, 2, 3, 1, 2, 2, 2, 1, 3, 3, 1, 2, 3, 4, 2, 4, 4, 3, 3],
                [1, 4, 4, 1, 2, 3, 1, 2, 1, 4, 2, 2, 3, 1, 1, 2, 2, 2, 2, 1]]
    for i in range(len(ceshi)):
        a = ceshi[i]
        if a == '22':
            j = 0
        elif a == '23':
            j = 1
        elif a == '24':
            j = 2
        else:
            j = 3
        objFilePath = "D:\\Filefolder\\UWM\\Second_seminar\\Independent_study\\pythonProject\\SS_for_AM\\Xdeformation\\G" + a + "_Xdeformation.txt"
        with open(objFilePath) as file:

            while 1:

                line = file.readline()
                if not line:
                    break
                strs = line.split("\t")
                if strs[0] == "Node Number":
                    continue
                else:
                    # print('strs:',strs)
                    if strs[2] == "2.5e-002":
                        xte.append((float(strs[1])))
                        zte.append((float(strs[3])))
                        a1te.append((float(layoutte[j][0])))
                        a2te.append((float(layoutte[j][1])))
                        a3te.append((float(layoutte[j][2])))
                        a4te.append((float(layoutte[j][3])))
                        a5te.append((float(layoutte[j][4])))
                        a6te.append((float(layoutte[j][5])))
                        a7te.append((float(layoutte[j][6])))
                        a8te.append((float(layoutte[j][7])))
                        a9te.append((float(layoutte[j][8])))
                        a10te.append((float(layoutte[j][9])))
                        a11te.append((float(layoutte[j][10])))
                        a12te.append((float(layoutte[j][11])))
                        a13te.append((float(layoutte[j][12])))
                        a14te.append((float(layoutte[j][13])))
                        a15te.append((float(layoutte[j][14])))
                        a16te.append((float(layoutte[j][15])))
                        a17te.append((float(layoutte[j][16])))
                        a18te.append((float(layoutte[j][17])))
                        a19te.append((float(layoutte[j][18])))
                        a20te.append((float(layoutte[j][19])))
                        Xdeformationte.append((float(strs[4])))
    a = np.arange(0, len(xte))
    random.shuffle(a)
    b = a[0:3600]
    xte = np.array(xte)
    x1te = xte[b]
    zte = np.array(zte)
    z1te = zte[b]
    a1te = np.array(a1te)
    a1ate = a1te[b]
    a2te = np.array(a2te)
    a2ate = a2te[b]
    a3te = np.array(a3te)
    a3ate = a3te[b]
    a4te = np.array(a4te)
    a4ate = a4te[b]
    a5te = np.array(a5te)
    a5ate = a5te[b]
    a6te = np.array(a6te)
    a6ate = a6te[b]
    a7te = np.array(a7te)
    a7ate = a7te[b]
    a8te = np.array(a8te)
    a8ate = a8te[b]
    a9te = np.array(a9te)
    a9ate = a9te[b]
    a10te = np.array(a10te)
    a10ate = a10te[b]
    a11te = np.array(a11te)
    a11ate = a11te[b]
    a12te = np.array(a12te)
    a12ate = a12te[b]
    a13te = np.array(a13te)
    a13ate = a13te[b]
    a14te = np.array(a14te)
    a14ate = a14te[b]
    a15te = np.array(a15te)
    a15ate = a15te[b]
    a16te = np.array(a16te)
    a16ate = a16te[b]
    a17te = np.array(a17te)
    a17ate = a17te[b]
    a18te = np.array(a18te)
    a18ate = a18te[b]
    a19te = np.array(a19te)
    a19ate = a19te[b]
    a20te = np.array(a20te)
    a20ate = a20te[b]
    Xdeformationte = np.array(Xdeformationte)
    Xdeformationtea = Xdeformationte[b]

    # normalize the training data------------------------
    x2 = ((x1 - (-0.015)) / (0.05 - (-0.015))).tolist()
    z2 = ((z1 - 0) / (0.04 - 0)).tolist()
    a1b = ((a1a - 1) / 3).tolist()
    a2b = ((a2a - 1) / 3).tolist()
    a3b = ((a3a - 1) / 3).tolist()
    a4b = ((a4a - 1) / 3).tolist()
    a5b = ((a5a - 1) / 3).tolist()
    a6b = ((a6a - 1) / 3).tolist()
    a7b = ((a7a - 1) / 3).tolist()
    a8b = ((a8a - 1) / 3).tolist()
    a9b = ((a9a - 1) / 3).tolist()
    a10b = ((a10a - 1) / 3).tolist()
    a11b = ((a11a - 1) / 3).tolist()
    a12b = ((a12a - 1) / 3).tolist()
    a13b = ((a13a - 1) / 3).tolist()
    a14b = ((a14a - 1) / 3).tolist()
    a15b = ((a15a - 1) / 3).tolist()
    a16b = ((a16a - 1) / 3).tolist()
    a17b = ((a17a - 1) / 3).tolist()
    a18b = ((a18a - 1) / 3).tolist()
    a19b = ((a19a - 1) / 3).tolist()
    a20b = ((a20a - 1) / 3).tolist()
    Xdeformationb = ((Xdeformationa - np.min(Xdeformationa)) / (np.max(Xdeformationa) - np.min(Xdeformationa))).tolist()
    # normalize the testing data
    x2te = ((x1te - (-0.015)) / (0.05 - (-0.015))).tolist()
    z2te = ((z1te - 0) / (0.04 - 0)).tolist()
    a1bte = ((a1ate - 1) / 3).tolist()
    a2bte = ((a2ate - 1) / 3).tolist()
    a3bte = ((a3ate - 1) / 3).tolist()
    a4bte = ((a4ate - 1) / 3).tolist()
    a5bte = ((a5ate - 1) / 3).tolist()
    a6bte = ((a6ate - 1) / 3).tolist()
    a7bte = ((a7ate - 1) / 3).tolist()
    a8bte = ((a8ate - 1) / 3).tolist()
    a9bte = ((a9ate - 1) / 3).tolist()
    a10bte = ((a10ate - 1) / 3).tolist()
    a11bte = ((a11ate - 1) / 3).tolist()
    a12bte = ((a12ate - 1) / 3).tolist()
    a13bte = ((a13ate - 1) / 3).tolist()
    a14bte = ((a14ate - 1) / 3).tolist()
    a15bte = ((a15ate - 1) / 3).tolist()
    a16bte = ((a16ate - 1) / 3).tolist()
    a17bte = ((a17ate - 1) / 3).tolist()
    a18bte = ((a18ate - 1) / 3).tolist()
    a19bte = ((a19ate - 1) / 3).tolist()
    a20bte = ((a20ate - 1) / 3).tolist()
    Xdeformationteb = ((Xdeformationtea - np.min(Xdeformationa)) / (np.max(Xdeformationa) - np.min(Xdeformationa))).tolist()

    # create dataset for training and testing, both including the data and target
    traindata = [[] for i in range(len(x2))]
    traintarget = []
    testdata = [[] for i in range(len(x2te))]
    testtarget = []
    for i in range(len(traindata)):
        traindata[i].append(x2[i])
        traindata[i].append(z2[i])
        traindata[i].append(a1b[i])
        traindata[i].append(a2b[i])
        traindata[i].append(a3b[i])
        traindata[i].append(a4b[i])
        traindata[i].append(a5b[i])
        traindata[i].append(a6b[i])
        traindata[i].append(a7b[i])
        traindata[i].append(a8b[i])
        traindata[i].append(a9b[i])
        traindata[i].append(a10b[i])
        traindata[i].append(a11b[i])
        traindata[i].append(a12b[i])
        traindata[i].append(a13b[i])
        traindata[i].append(a14b[i])
        traindata[i].append(a15b[i])
        traindata[i].append(a16b[i])
        traindata[i].append(a17b[i])
        traindata[i].append(a18b[i])
        traindata[i].append(a19b[i])
        traindata[i].append(a20b[i])

    for i in range(len(Xdeformationb)):
        traintarget.append(Xdeformationb[i])

    for i in range(len(testdata)):
        testdata[i].append(x2te[i])
        testdata[i].append(z2te[i])
        testdata[i].append(a1bte[i])
        testdata[i].append(a2bte[i])
        testdata[i].append(a3bte[i])
        testdata[i].append(a4bte[i])
        testdata[i].append(a5bte[i])
        testdata[i].append(a6bte[i])
        testdata[i].append(a7bte[i])
        testdata[i].append(a8bte[i])
        testdata[i].append(a9bte[i])
        testdata[i].append(a10bte[i])
        testdata[i].append(a11bte[i])
        testdata[i].append(a12bte[i])
        testdata[i].append(a13bte[i])
        testdata[i].append(a14bte[i])
        testdata[i].append(a15bte[i])
        testdata[i].append(a16bte[i])
        testdata[i].append(a17bte[i])
        testdata[i].append(a18bte[i])
        testdata[i].append(a19bte[i])
        testdata[i].append(a20bte[i])

    for i in range(len(Xdeformationteb)):
        testtarget.append(Xdeformationteb[i])

    #print('traindata:', traindata,"/n")
    #print('traintarget:', traintarget, "/n")
    #print('testdata:',testdata, "/n")
    #print('testtarget:', testtarget, "/n")

    return traindata, traintarget, testdata, testtarget, np.max(Xdeformationa),\
        np.min(Xdeformationa), np.max(Xdeformationtea), np.min(Xdeformationtea)

# create the train dataloader and test dataloader
xunlian = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '13', '14', '15', '16', '17', '19', '20', '21']
ceshi = ['25']
traindata, traintarget, testdata, testtarget,  maxXdeformationa, minXdeformationa, maxXdeformationtea, minXdeformationtea\
    = readfile(xunlian, ceshi)
my_train_dataset = MyDataset(torch.Tensor(traindata), torch.Tensor(traintarget))
train_dataloader = torch.utils.data.DataLoader(my_train_dataset, batch_size=5,shuffle=True)
my_test_dataset = MyDataset(torch.Tensor(testdata), torch.Tensor(testtarget))
test_dataloader = torch.utils.data.DataLoader(my_test_dataset, batch_size=5)


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(22, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# to train a model, we need a loss function and an optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        y = y.view(5,1)
        test_loss += loss_fn(pred, y).item()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_loss /= num_batches
    print(f"Train Error: \n  Avg loss: {test_loss:>8f} \n")
    """if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")"""
    return test_loss

# We also check the model’s performance against the test dataset to ensure it is learning
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.view(5, 1)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    #correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")
    return test_loss
"""
# iterate the training and testing
epochs = 10000
trainL = []
testL = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    trainL.append(train_loss)
    test_loss = test(test_dataloader, model, loss_fn)
    testL.append(test_loss)
print("Done!")
# print('trainL:',trainL)
# print('testL:',testL)

# saving the model
torch.save(model.state_dict(), "MLP_SS_model_5.pth")
print("Saved PyTorch Model State to MLP_SS_model_5.pth")
"""
# loading models
model = NeuralNetwork()
start1 = time.time()
model.load_state_dict(torch.load("MLP_SS_model_3.pth"))
end1 = time.time()
print("Model_load_time: ", end1-start1)  # Time the load modeling process

# to predict the model
model.eval()
num_batches = len(test_dataloader)
test_loss, correct = 0, 0
predict = []
with torch.no_grad():
    """
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        model.to(device)
        pred = model(X)
        y = y.view(5, 1)
        pred1 = pred.tolist()
        predict.append(pred1)
        test_loss += loss_fn(pred, y).item()
        #correct += (pred.argmax(1) == y).type(torch.float).sum().item()"""

    start2 = time.time()
    testy = model(torch.Tensor(testdata))
    end2 = time.time()
    print("Model_Predict_time: ", end2 - start2)  # Time the predicting modeling process

    test_loss += loss_fn(testy, torch.Tensor(testtarget)).item()
    print(f"Test Loss: {test_loss:.4f}")
    predict = testy.tolist()
    predict = np.array(predict).reshape(3600)


predict = np.array(predict)
predict = predict.reshape(3600)
#print('predict: ',predict)
test_loss /= num_batches
#correct /= size
#print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")


# function for transforming the normalized data back to original status
def de_normal(data, maxXdeformationa, minXdeformationa):
    DATA = ((data - minXdeformationa) * (maxXdeformationa - minXdeformationa)).tolist()

    return DATA


# transform the data
predict = de_normal(predict, maxXdeformationa, minXdeformationa)
testtarget = de_normal(testtarget, maxXdeformationa, minXdeformationa)
"""
# plot the loss curve of training and testing
x = np.arange(epochs)
#print('x:',x)
plt.plot(x, trainL)
plt.plot(x, testL)
plt.legend(['training loss','testing loss'])"""

# plot the prediction and original figure
plt.figure()
testdata = np.array(testdata)
plt.scatter(testdata[:,0],testdata[:,1],s=60, c=predict)
plt.colorbar()
plt.title('predict')

plt.figure()
plt.scatter(testdata[:,0],testdata[:,1],s=60, c=testtarget)
plt.colorbar()
plt.title('original')
plt.show()

# print the R2_score
predict = np.array(predict)
testtarget = np.array(testtarget)
print('r2_score: %.5f'
      % r2_score(testtarget, predict))
