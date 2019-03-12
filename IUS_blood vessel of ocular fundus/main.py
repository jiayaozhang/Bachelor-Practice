import numpy as np
import torch
import torch.nn as nn
import dataProcess
from torch.autograd import Variable
import model
import visdom
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from scipy import interp
def epochTrain(net,optim,batch,picNames):
    net.train()
    trainLoss = np.array([])
    dataLoader = torch.utils.data.DataLoader(dataProcess.MyDataLoader(picNames),batch_size=batch,shuffle=True,drop_last=False)
    for data,label in dataLoader:
        batchLoss = 0
        inputs = Variable(data).cuda()
        label = Variable(label.float()).cuda()
        pred = net(inputs)
        optim.zero_grad()
        pred = pred.view(-1)
        batchLoss -= torch.matmul(label,torch.log(pred+0.000000001))
        batchLoss -= torch.matmul((1-label),torch.log(1-pred+0.00000001))
        batchLoss.backward()
        optim.step()
        trainLoss = np.append(trainLoss,batchLoss.data.cpu().numpy())
    return trainLoss.mean()

def score(pred,label):
    pred = pred.data.cpu().numpy()
    pred = pred.reshape(-1)
    label = label.data.cpu().numpy()
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    predPos = pred.sum()
    labelPos = label.sum()
    recall = ((pred+label)==2).sum()
    precision = recall*1.0 / (predPos+0.0000000001)
    recallRate = recall*1.0 / (labelPos+0.000000001)
    f1 = 2*precision*recallRate / (precision+recallRate+0.0000000001)
    return precision,recallRate,f1

def rocPlot(pred,label):
    predTemp = pred.data.cpu().numpy().reshape(-1)
    labelTemp = label.data.cpu().numpy().reshape(-1).astype(int)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1,100)
    all_tpr = []
    fpr,tpr,thresh = roc_curve(labelTemp,predTemp,pos_label=1)
    #print fpr,'\n',tpr,'\n',thresh
    mean_tpr = interp(mean_fpr,fpr,tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr,tpr)
    print roc_auc
    plt.figure(figsize=(10,10))
    plt.plot(fpr,tpr,lw=1,label='AUC=%0.2f' % roc_auc)
    plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label='Luck')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('./picture/ROC.jpg')
    plt.close('all')

def epochTest(net,batch,picNames):
    net.eval()
    testLoss = np.array([])
    dataLoader = torch.utils.data.DataLoader(dataProcess.MyDataLoader(picNames),batch_size=batch,shuffle=True,drop_last=False)
    for data,label in dataLoader:
        batchLoss = 0
        inputs = Variable(data).cuda()
        label = Variable(label.float()).cuda()
        pred = net(inputs)
        batchLoss -= torch.matmul(label,torch.log(pred+0.00000000001))
        batchLoss -= torch.matmul((1-label),torch.log(1-pred+0.0000000001))
        testLoss = np.append(testLoss,batchLoss.data.cpu().numpy())
    rocPlot(pred,label)
    return [testLoss.mean(),score(pred,label)]

def mainDk(EPOCH,lr,batch,trainFiles,testFiles):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    vis = visdom.Visdom(env='dengkai',port=8102)
    visWindow = [None,None,None,None,None]
    opts = [dict(title='trainLoss',ylabel='Loss',xLabel='epoch'),
            dict(title='testLoss',ylabel='Loss',xLabel='epoch'),
            dict(title='precision',ylabel='precision',xLabel='epoch'),
            dict(title='recall',ylabel='recallRate',xLabel='epoch'),
            dict(title='f1',ylabel='score',xLabel='epoch')]
    trainLoss,testLoss = [],[]
    pre,recall,f1 = [],[],[]
    myNet = model.ResNet(model.ResidualBlock,[1,1,1]).cuda()
    optim = torch.optim.Adam(myNet.parameters(),lr=lr)
    for epoch in range(EPOCH):
        trainLossTemp = epochTrain(myNet,optim,batch,trainFiles)
        trainLoss.append(trainLossTemp)
        testScore = epochTest(myNet,32,testFiles)
        print 'epoch:',epoch,testScore
        testLoss.append(testScore[0])
        pre.append(testScore[1][0])
        recall.append(testScore[1][1])
        f1.append(testScore[1][2])
        if epoch % 9 == 0:
            lr /=10
            optim = torch.optim.Adam(myNet.parameters(),lr=lr)
        torch.save(myNet.state_dict(),'./net_state/net_origin.pkl')
        xAxis = np.arange(epoch+1)
        yAxis = [trainLoss,testLoss,pre,recall,f1]
        for x in range(len(visWindow)):
            if visWindow[x] is None:
                visWindow[x] = vis.line(X=np.array(xAxis),Y=np.array(yAxis[x]),opts=opts[x])
            else:
                vis.line(X=np.array(xAxis),Y=np.array(yAxis[x]),win=visWindow[x],update='replace')

if __name__=='__main__':
    
    picSample = np.arange(79)
    np.random.shuffle(picSample)
    trainNum = 63
    picPaths = []
    for main,sub,files in os.walk('./data/pos'):
        for fileTemp in files:
            if '.bmp' in fileTemp:
                picPaths.append(os.path.join(main,fileTemp))
    picPaths = np.array(picPaths)

    trainPaths = picPaths[picSample[:trainNum]]
    testPaths = picPaths[picSample[trainNum:]]

    picSample = np.arange(79)
    np.random.shuffle(picSample)
    picPaths = []
    for main,sub,files in os.walk('./data/neg'):
        for fileTemp in files:
            if '.bmp' in fileTemp:
                picPaths.append(os.path.join(main,fileTemp))
    picPaths = np.array(picPaths)

    trainPaths = np.append(trainPaths,picPaths[picSample[:trainNum]])
    testPaths = np.append(testPaths,picPaths[picSample[trainNum:]])
    EPOCH = 100
    lr = 0.0001
    batch = 30
    mainDk(EPOCH,lr,batch,trainPaths,testPaths)

