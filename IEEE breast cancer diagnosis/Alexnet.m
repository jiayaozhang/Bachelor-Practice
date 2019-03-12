%get train data
mama_ds = imageDatastore('photoacoustic_data_crop224','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(mama_ds,0.8,'randomized');
numClasses = numel(categories(mama_ds.Labels));
%create a network by modifying Alexnet
% net = alexnet;
net = vgg19;
% net = resnet50;
maxepoch = 12;
minibatchsize = 20;

layers = net.Layers;
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;
%set training option
options = trainingOptions('sgdm','InitialLearnRate', 0.001,...
    'ValidationData',testImgs, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false ,...
    'Plots','training-progress',...
    'MaxEpochs',maxepoch,...
    'MiniBatchSize',minibatchsize);
%perform training
[mamanet,info] = trainNetwork(trainImgs, layers, options);
%pred
testpreds = classify(mamanet,testImgs);
YValidation = testImgs.Labels;


%change name to number
len = length(YValidation);
test = zeros(len,1);
pred = zeros(len,1);
TP = 0;TN = 0;FP = 0;FN = 0;
for i = 1:len
    if YValidation(i) == 'cancer'
        test(i) = 1;
    else
         test(i) = 0;
    end
    if testpreds(i) == 'cancer'
        pred(i) = 1;
    else
         pred(i) = 0;
    end
    if YValidation(i) == 'cancer' && testpreds(i) == 'cancer'
        TP = TP + 1;
    end
    if YValidation(i) == 'normal' && testpreds(i) == 'normal'
        TN = TN + 1;
    end
    if YValidation(i) == 'normal' && testpreds(i) == 'cancer'
        FP = FP + 1;
    end
    if YValidation(i) == 'cancer' && testpreds(i) == 'normal'
        FN = FN + 1;
    end
end
% Auc = AUC(pred,test)
Auc = AUC(test,pred)
Sensitivity = TP/(TP + FN)
Speciality = TN/(TN + FP)
Accuracy = mean(testpreds == YValidation)




