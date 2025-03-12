%% Deep Learning Image Classification Project
% This script trains a CNN model to classify images using MATLAB's Deep Learning Toolbox.

clear; clc; close all;

% Load dataset
[dataTrain, dataTest] = loadData();

% Train model
trainedNet = trainModel(dataTrain);

% Evaluate model
evaluateModel(trainedNet, dataTest);

% Save trained model
save('trainedModel.mat', 'trainedNet');

disp('Model training and evaluation complete.');

%% loadData.m: Function to Load and Preprocess Data
function [dataTrain, dataTest] = loadData()
    % Load dataset (example: use MATLAB's ImageDatastore for folder-based images)
    datasetPath = 'dataset'; % Change this to your dataset folder
    imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    % Split dataset
    [dataTrain, dataTest] = splitEachLabel(imds, 0.8, 'randomized');
    
    % Resize images to match CNN input size
    inputSize = [227 227];
    dataTrain.ReadFcn = @(x) imresize(imread(x), inputSize);
    dataTest.ReadFcn = @(x) imresize(imread(x), inputSize);
end

%% trainModel.m: Function to Train CNN
function trainedNet = trainModel(dataTrain)
    % Define CNN architecture
    layers = [ 
        imageInputLayer([227 227 3])
        convolution2dLayer(3, 16, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        fullyConnectedLayer(2) % Change based on number of classes
        softmaxLayer
        classificationLayer];
    
    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 32, ...
        'ValidationData', dataTrain, ...
        'ValidationFrequency', 5, ...
        'Verbose', true, ...
        'Plots', 'training-progress');
    
    % Train the network
    trainedNet = trainNetwork(dataTrain, layers, options);
end

%% evaluateModel.m: Function to Evaluate Model
function evaluateModel(trainedNet, dataTest)
    % Classify test images
    predictedLabels = classify(trainedNet, dataTest);
    actualLabels = dataTest.Labels;
    
    % Compute accuracy
    accuracy = sum(predictedLabels == actualLabels) / numel(actualLabels);
    fprintf('Model Accuracy: %.2f%%\n', accuracy * 100);
    
    % Confusion Matrix
    figure;
    confusionchart(actualLabels, predictedLabels);
    title('Confusion Matrix');
end

