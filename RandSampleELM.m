function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = RandSampleELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction,C)

% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
% Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
%


%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;
KNN_Par = 5;
%%%%%%%%%%% Load training dataset
train_data=TrainingData_File;
clear TrainingData_File;
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=TestingData_File;
clear TestingData_File;
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    Max_Perclass_Num = 1; tmp_Max_Perclass_Num = 1;
    %%%%%%%%%%%% Preprocessing the data of classification
%     sorted_target=sort(cat(2,T,TV.T),2);
    sorted_target=sort(T,2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:NumberofTrainingData%+NumberofTestingData
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
            if tmp_Max_Perclass_Num > Max_Perclass_Num
                Max_Perclass_Num = tmp_Max_Perclass_Num;
            end
            tmp_Max_Perclass_Num = 1;
        else
            tmp_Max_Perclass_Num = tmp_Max_Perclass_Num + 1;
        end
    end
    number_class=j;
    %%% Save the classes
    Class_Data = zeros(number_class,Max_Perclass_Num) * 1 / 0;
    Class_Data_Flag = zeros(number_class,1);
    %%%
    NumberofOutputNeurons=number_class;
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
        Class_Data_Flag(j,1) = Class_Data_Flag(j,1) + 1;
        Class_Data(j,Class_Data_Flag(j,1)) = i;
    end
    T=temp_T*2-1;
    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
end                                                 %   end if of Elm_Type
%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;
%%%% Random generated sample divisions
% figure
% plot(P(1,:)',P(2,:)','.');
InputWeight = zeros(NumberofHiddenNeurons,NumberofInputNeurons);
BiasofHiddenNeurons=zeros(NumberofHiddenNeurons,1);
InputWeightIndice = zeros(NumberofHiddenNeurons,2);
Flag = 0;
i = 1;
while i < NumberofHiddenNeurons + 1
    
    %%% Random choose two classes
    Class_1 = 1; Class_2 = 1;
    %%% Select different sample class pairs
    %%% to different samples
    InputWeightIndice(i,:) = [Class_1, Class_2];
    %%% Select two different classes separatly
    Class_1_Label = 1; Class_2_Label = 1;
    while Class_1_Label == Class_2_Label
        Class_1 = randi(NumberofTrainingData,1); Class_2 = randi(NumberofTrainingData,1);
        Class_1_Label = find(T(:,Class_1)'+1,2); Class_2_Label = find(T(:,Class_2)'+1,2);
    end
    if Class_1_Label > Class_2_Label
        tmp = Class_1; Class_1 = Class_2; Class_2 = tmp;
    end
    %%% Delete the small different weights
    j = 1;
    while j < i
        %if (abs(dot(InputWeight(j,:),P(:,Class_1)'-P(:,Class_2)')/ norm(InputWeight(j,:)) / norm(P(:,Class_1)'-P(:,Class_2)'))>1-1/(Flag*5)) || ...
        if dot(P(:,Class_1)-P(:,Class_2),P(:,Class_1)-P(:,Class_2)) < 1/(Flag*5)
            Flag = Flag + 1;
            break;
        end
        j = j +1;
    end
    if j == i
        InputWeight(i,:) = P(:,Class_1) - P(:,Class_2);
        NormSquareInputWeight = dot(InputWeight(i,:),InputWeight(i,:));
        InputWeight(i,:) = InputWeight(i,:)./(NormSquareInputWeight);
        %BiasofHiddenNeurons(i,1) = -(dot(P(:,Class_2),P(:,Class_2))-dot(P(:,Class_1),P(:,Class_1)))/2/NormSquareInputWeight;
        %BiasofHiddenNeurons(i,1) = (dot(P(:,Class_2),P(:,Class_2))-dot(P(:,Class_1),P(:,Class_1)))/2/NormSquareInputWeight;
        i = i + 1;
    end
end

% for i = 1:NumberofHiddenNeurons
%     %%% Random choose two classes
%     Class_1 = 1; Class_2 = 1;
%     while find(T(:,Class_1)'+1,2) == find(T(:,Class_2)'+1,2)
%         Class_1 = randi(NumberofTrainingData,1); Class_2 = randi(NumberofTrainingData,1);
%     end
%     if find(T(:,Class_1)'+1,2) > find(T(:,Class_2)'+1,2)
%         tmp = Class_1; Class_1 = Class_2; Class_2 = tmp;
%     end
%     InputWeight(i,:) = P(:,Class_1) - P(:,Class_2);
%     InputWeight(i,:) = InputWeight(i,:)./sqrt(sum(InputWeight(i,:).^2));
% end

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
% InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
% BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
%BiasofHiddenNeurons=-sum(InputWeight,2)./2;
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
if  C == 10 ^ 100
    OutputWeight=pinv(H') * T';                        % implementation without regularization factor 
%OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   
%implementation; one can set regularizaiton factor C properly in classification applications 
else
    OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      
%implementation; one can set regularizaiton factor C properly in classification applications
end

end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case
end
clear H;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
clear tempH_test;
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
clear H_test;
if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);  
end


% %%%%%%%%%%% Draw the decision boundary
% Decision_Data = [];
% for i = -1:0.01:1
%     for j = -1:0.01:1
%        Decision_Data = [Decision_Data,[i;j]];
%     end
% end
% Decision_Data_test=InputWeight*Decision_Data;
% NumberofTestingData = size(Decision_Data_test,2);            
% ind=ones(1,NumberofTestingData);
% BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
% Decision_Data_tempH_test=Decision_Data_test + BiasMatrix;
% switch lower(ActivationFunction)
%     case {'sig','sigmoid'}
%         %%%%%%%% Sigmoid 
%         Decision_Data_test_H_test = 1 ./ (1 + exp(-Decision_Data_tempH_test));
%     case {'sin','sine'}
%         %%%%%%%% Sine
%         Decision_Data_test_H_test = sin(Decision_Data_tempH_test);        
%     case {'hardlim'}
%         %%%%%%%% Hard Limit
%         Decision_Data_test_H_test = hardlim(Decision_Data_tempH_test);        
%     case {'tribas'}
%         %%%%%%%% Triangular basis function
%         Decision_Data_test_H_test = tribas(Decision_Data_tempH_test);        
%     case {'radbas'}
%         %%%%%%%% Radial basis function
%         Decision_Data_test_H_test = radbas(Decision_Data_tempH_test);        
%         %%%%%%%% More activation functions can be added here
% end
% clear Decision_Data_tempH_test;
% Decision_Data_TY=(Decision_Data_test_H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
% clear Decision_Data_test_H_test;
% Decision_Data_Indice = [];
% for i = 1 : NumberofTestingData
%     if Decision_Data_TY(1,i) < 0.005 && Decision_Data_TY(1,i) > -0.005
%         Decision_Data_Indice = [Decision_Data_Indice,i];
%     end
% end
% hold on
% plot(Decision_Data(1,Decision_Data_Indice),Decision_Data(2,Decision_Data_Indice),'r.');
% saveas(gcf,strcat('RandomKNNRandBias',num2str(NumberofHiddenNeurons),'.fig'));
% close all;
