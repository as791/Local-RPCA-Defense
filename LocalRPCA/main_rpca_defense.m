% Copyright (C) 2021, Aryaman Sinha
clc;
clear;
close all;
%% initialize num of extra copies and classifier
net = vgg16;
%% read dir and extract filenames of the data
database_dir = '../vgg/cw_l2/';

files = dir(database_dir);
directoryNames = {files.name};
 orig_input = containers.Map('KeyType','uint32','ValueType','any');
adv_input = containers.Map('KeyType','uint32','ValueType','any');
num1=0;num2=0;
for f = directoryNames
    f1 = f{1};
    if contains(f1,'_')
        data = strsplit(f1,'_');
        if contains(f1,'orig')
             orig_input(num1+1)=f1;
             num1=num1+1;
         end
        if contains(f1,'adv')
            adv_input(num2+1)=f1;
            num2=num2+1;
        end
    end
end

%% testing for adv input
cnt=0;
min_psnr=100;
num = size(adv_input,1); 
for i=1:num
    % load image from .mat files
    if isKey(adv_input,i)
    file1 = adv_input(i);
    file2 = orig_input(i);
    test_img = double(load(strcat(database_dir,file1)).adv)*255;
    orig_img = double(load(strcat(database_dir,file2)).orig)*255;
    
     % Checking RMSE value
%     dist = (test_img-orig_img)/255;
%     rmse = sqrt(sum(dist.*dist,'all')/(224*224*3));
%     fprintf('rmse:%.4f\n',rmse);
                 
    tol = 1e-6;
    max_iter = 1000; 
    n = 7;
    lambda = 1/sqrt(n);                
     mu = 0.02;
    
    L1=test_img(:,:,1);
    L2=test_img(:,:,2);
    L3=test_img(:,:,3);

%     [L1, S1] = RobustPCA(L1, lambda, mu, tol,max_iter);
%     [L2, S2] = RobustPCA(L2, lambda, mu, tol,max_iter);
%     [L3, S3] = RobustPCA(L3, lambda, mu, tol,max_iter);
     [L1,S1] = blockRPCA(L1,n,lambda,mu);
     [L2,S2] = blockRPCA(L2,n,lambda,mu);
     [L3,S3] = blockRPCA(L3,n,lambda,mu);

    img(:,:,1)=L1;
    img(:,:,2)=L2; 
    img(:,:,3)=L3;
    
    img = double(uint8(img));
    
    img = wavelet_denoising(img,0.04);
   
    PSNR = psnr(img,orig_img,255);
    
%      display_predictions(net,orig_img);
%      display_predictions(net,test_img);
%      display_predictions(net,img);
%      imshow(img/255);figure;
%      imshow(test_img/255);figure;
%      imshow((test_img-img)/255,[0 1]);
      
    if classify(net,img) == classify(net,orig_img)
        cnt = cnt+1;
        min_psnr = min(min_psnr,PSNR);
    end 
    end
end
fprintf('Success Rate: %.4f\n',cnt*100/num);
fprintf('Min PSNR(dB): %.4f\n',min_psnr);


%% testing for orig input
cnt = 0;
min_psnr=100;
num = size(orig_input,1); 
for i=1:num
    % load image from .mat file
    file1 = orig_input(i);
    orig_img = double(load(strcat(database_dir,file1)).orig)*255;
   
    n = 7;
    tol = 1e-6;
     lambda = 1/sqrt(n);  
    max_iter=1000;
    mu = 0.02;
    
    L1=orig_img(:,:,1);
    L2=orig_img(:,:,2);
    L3=orig_img(:,:,3);
    
%      [L1, S1] = RobustPCA(L1, lambda, mu, tol, max_iter);
%      [L2, S2] = RobustPCA(L2, lambda, mu, tol,max_iter);
%      [L3, S3] = RobustPCA(L3, lambda, mu, tol,max_iter);
     [L1,S1] = blockRPCA(L1,n,lambda,mu);
     [L2,S2] = blockRPCA(L2,n,lambda,mu);
     [L3,S3] = blockRPCA(L3,n,lambda,mu);
    
    img(:,:,1)=L1;
    img(:,:,2)=L2;
    img(:,:,3)=L3;

    img = double(uint8(img));
    
    img = wavelet_denoising(img,0.04);
    
    PSNR = psnr(img,orig_img,255);

    if classify(net,img) == classify(net,orig_img)
        cnt = cnt+1;
        min_psnr = min(min_psnr,PSNR);
    end
end
fprintf('Accuracy: %.4f\n',cnt*100/num);
fprintf('Min PSNR(dB): %.4f\n',min_psnr);
