%%%%%%%%%training on dataset\
clc;
clear all;
close all;

folder = 'G:\project\dataset\folder1\train\pos';
files = dir(fullfile(folder, '*.jpeg'));
% file1  = numel(files)
folder2='G:\project\dataset\folder1\train\neg';
files2 = dir(fullfile(folder2,'*.jpg'));
% file2 = numel(files2)
totalNumofImages = numel(files)+numel(files2);
count1 = 0;
labels = zeros(totalNumofImages, 1);

[rows,cols,width]=size(imread([folder '/' files(1).name]));
cellsize = 8;
blockSize=4;
numBins =9;
num_cell_row = floor(rows/cellsize);
num_cell_cols = floor(cols/cellsize);
descriptorSize = (num_cell_row-1)*(num_cell_cols-1)*blockSize*numBins;
features = zeros(totalNumofImages, descriptorSize);

for k = 1:numel(files)%totalNumofImages
    
    count1 = count1 + 1;
    filename = [folder '/' files(k).name];
    im = imread(filename);
    [features,labels] = hog(im,count1,1,features,labels);
end

count2=0;
for k = 1:numel(files2)%1:numel(files)%totalNumofImages
    
    count2 = count2 + 1;
    filename = [folder2 '/' files2(k).name];
    im = imread(filename);
    im = imresize(im,[128 64]);
    total = count1 + count2;
[features,labels] = hog(im,total,-1,features,labels);
end
SVMSTRUCT = svmtrain(features, labels);

folder3 = 'G:\project\dataset\folder1\test\pos';
files3 = dir(fullfile(folder3, '*.jpeg'));
% file1  = numel(files3)
folder4='G:\project\dataset\folder1\test\neg';
files4 = dir(fullfile(folder4,'*.jpg'));

% test_img = imread('G:\project\dataset\folder1\test\pos\per00080.jpeg');
% [features1,labels1] = hog(test_img,1,1);
% lbl = svmclassify(SVMSTRUCT,features1);
% file2 = numel(files4)

cnt = 0;
features1 = zeros(1, descriptorSize);
for i = 1 : numel(files3)
    filename = [folder3 '/' files3(i).name];
    test_img = imread(filename);
    [features1,labels1] = hog(test_img,1,1,features1,1);
    lbl = svmclassify(SVMSTRUCT,features1);
    if(lbl==1)
        cnt = cnt+1;
    end
end

cnt1 = 0;
features2 = zeros(1, descriptorSize);
for i = 1 : numel(files4)
    filename = [folder4 '/' files4(i).name];
    test_img = imread(filename);
    test_img = imresize(test_img,[128 64]);
    [features2,labels2] = hog(test_img,1,1,features2,1);
    lbl = svmclassify(SVMSTRUCT,features2);
    if(lbl==1)
        cnt1 = cnt1+1;
    end
end



    




