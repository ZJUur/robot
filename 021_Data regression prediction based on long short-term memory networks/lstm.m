%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��������  ��4����1��� �ܹ�103��������
res = xlsread('800 1200.xls');

%%  ����ѵ�����Ͳ��Լ�
temp = randperm(300);%300��������������������������ݼ�������һ���������� temp=1��1��300  �����ҵķ�ʽ

P_train = res(temp(1: 200), 2: 5)';%ѵ��������Ϊ���Һ��ǰ200�е�2-5��
T_train = res(temp(1: 200), 1)';%ѵ�������Ϊ��һ��
M = size(P_train, 2);%��ת�ã��в���������Ŀ
fprintf('the value of M is%6.2f\n',M)
P_test = res(temp(201: end), 2: 5)';%���Լ�����
T_test = res(temp(201: end), 1)';%���Լ����
N = size(P_test, 2);

%%  ���ݹ�һ��
[P_train, ps_input] = mapminmax(P_train, 0, 1);%ѵ���������һ����01֮��
P_test = mapminmax('apply', P_test, ps_input);%���Լ������һ����01֮��

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  ����ƽ�̣������繤����Ҫ��4ά���ݣ����������7 1 1  M
P_train =  double(reshape(P_train, 4, 1, 1, M));%7 1 1  ��ʾ7*1*1������ MΪѵ����������Ŀ  
P_test  =  double(reshape(P_test , 4, 1, 1, N));

t_train = t_train';%�������ת�ã���ʽҪ��
t_test  = t_test' ;



%%  ���ݸ�ʽת�� ת����Ԫ���������ʽ
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  ����ģ��
layers = [
    sequenceInputLayer(4)               % ��������㣬���ݼ�������д����
    
    lstmLayer(4, 'OutputMode', 'last')  % LSTM�㣬4�����ز㵥Ԫ
    reluLayer                           % Relu�����
    
    fullyConnectedLayer(1)              % ȫ���Ӳ㣬��������㣬1���������1
    regressionLayer];                   % �ع��
 
%%  ��������
options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨������sgdm
    'MiniBatchSize', 30, ...               % ������С��ÿ���ö�������ѵ��������������80�����ݣ�ÿ����30����ѵ�����Σ�ʣ�µ�20����Ҫ�ˡ�
    'MaxEpochs', 1200, ...                 % ������������ѵ��1200�� 1200*2=2400��
    'InitialLearnRate', 1e-2, ...          % ��ʼѧϰ��Ϊ0.01
    'LearnRateSchedule', 'piecewise', ...  % ѧϰ���½�
    'LearnRateDropFactor', 0.5, ...        % ѧϰ���½�����
    'LearnRateDropPeriod', 800, ...        % ���� 800 ��ѵ���� ѧϰ��Ϊ 0.01 * 0.5
    'Shuffle', 'every-epoch', ...          % ÿ��ѵ���������ݼ�
    'Plots', 'training-progress', ...      % ��������
    'Verbose', false);%�ر���������ʾ����Ϣ

%%  ѵ��ģ��
net = trainNetwork(p_train, t_train, layers, options);

%%  ����Ԥ��
t_sim1 = predict(net, p_train);%ѵ��������Ԥ�⣬����ģ��ѵ�����
t_sim2 = predict(net, p_test );%���Լ�����Ԥ�⣬����ģ��ѵ�����

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  ���������
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%%  �鿴����ṹ
analyzeNetwork(net)

%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  ���ָ�����
%  R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;

disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2)])

save('800 1200.mat','net')
disp('�����ѱ���')
save('800 1200para.mat','ps_input','ps_output')
disp('��һ�������ѱ���')