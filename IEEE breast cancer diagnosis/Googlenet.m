%get train data
mama_ds = imageDatastore('photoacoustic_data_crop224','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(mama_ds,0.8);
numClasses = numel(categories(mama_ds.Labels));
%create a network by modifying Alexnet
net = googlenet;
lgraph = layerGraph(net);
% figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
% plot(lgraph);
lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])
%%%%%%%%%% Freeze Initial Layers
% layers = lgraph.Layers;
% connections = lgraph.Connections;

% layers(1:110) = freezeWeights(layers(1:110));
% lgraph = createLgraphUsingConnections(layers,connections);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',testImgs, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false ,...
    'Plots','training-progress');

mamanet = trainNetwork(trainImgs,lgraph,options);

[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == testImgs.Labels)


% %set training option
% options = trainingOptions('sgdm','InitialLearnRate', 0.001,'MaxEpochs',8,'Plots','training-progress');
% %perform training
% [mamanet,info] = trainNetwork(trainImgs, layers, options);
% %pred
% testpreds = classify(mamanet,testImgs);
% YValidation = testImgs.Labels;
% accuracy = mean(testpreds == YValidation)


