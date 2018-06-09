function test_play
clc; clear; close all;  % clear
global w0 wn;
structData = dir('*.mp3'); % get all files in directory with mp3 extension
Names = structData.name; 
len = length(structData); % receiving the number of elements of the directory with mp3 extension

MaxLenData = 44100 * 10; % the length of the file for training is 10 seconds
XDATA_L_learn = []; XDATA_R_learn = []; % allocation of place for data for training
for i = 1:2 %  read the first two files for learning
    way = strcat(structData(i).folder,'/',structData(i).name); % have created a complete path to the file
    [Y, ~] = audioread(way, [1 MaxLenData]); %  read the first 10 seconds of the file
    XDATA_L_learn = [XDATA_L_learn Y(:,1)]; % recorded the left stream of the read file
    XDATA_R_learn = [XDATA_R_learn Y(:,2)]; % recorded the right stream of the read file
end
way = strcat(structData(3).folder,'/',structData(3).name); % read the third file for learning
[Y, ~] = audioread(way, [1 MaxLenData]); % read 3rd study file

YDATA_L_learn = []; YDATA_R_learn = []; % form data for the output of the neural network
YDATA_L_learn = [YDATA_L_learn Y(:,1)]; % recorded left flow
YDATA_R_learn = [YDATA_R_learn Y(:,2)]; % recorded the right flow

XDATA_learn = 0.5*(XDATA_L_learn + XDATA_R_learn); % is the arithmetic mean of the X data for training
YDATA_learn = 0.5*(YDATA_L_learn + YDATA_R_learn); % is the arithmetic average of the data for learning
ind = 1:300:MaxLenData; % for training we take every 300th value
x = XDATA_learn(ind,:); % recorded arguments x
y = YDATA_learn(ind);   % recorded the value of y
BP_Algo(x,y); % caused by the training of the neural network

%% create a file based on the model (check the learning NN)
XDATA_L_test = []; XDATA_R_test = [];
for i=4:5 % read the files to check the model's performance
    way = strcat(structData(i).folder,'/',structData(i).name); % path to file
    [Y, ~] = audioread(way, [1 MaxLenData]); % read the file
    XDATA_L_test = [XDATA_L_test Y(:,1)]; % added data X
    XDATA_R_test = [XDATA_R_test Y(:,2)]; % added data to

Y_L_test = []; Y_R_test = []; % formation of arrays for results
w0 = w0/norm(w0); w0 = wn/norm(wn);
s0 = mean(mean(abs(w0))); sn = mean(abs(wn));
for i=1:length(XDATA_R_test(:,1))
    x = XDATA_L_test(i,:); % Left Flow (Input)

    Y_L_test = [Y_L_test; s0*x(1) + sn*x(2)]; % recorded the result of the model to the left stream
    x = XDATA_R_test(i,:); % right stream (input)
    Y_R_test = [Y_R_test; s0*x(1) + sn*x(2)]; % recorded the result of the model to the right stream
    if mod(i,5000)==0
        fprintf('i = %d\n',i);
    end
end
% record the result
audiowrite('Rez.mp4',3*[Y_L_test Y_R_test],44100);