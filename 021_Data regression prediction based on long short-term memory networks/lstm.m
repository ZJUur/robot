%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据  （4输入1输出 总共103个样本）
res = xlsread('800 1200.xls');

%%  划分训练集和测试集
temp = randperm(300);%300个样本，生成乱序数组打乱数据集，生成一个乱序数组 temp=1：1：300  不打乱的方式

P_train = res(temp(1: 200), 2: 5)';%训练集输入为打乱后的前200行第2-5列
T_train = res(temp(1: 200), 1)';%训练集输出为第一列
M = size(P_train, 2);%有转置，列才是样本数目
fprintf('the value of M is%6.2f\n',M)
P_test = res(temp(201: end), 2: 5)';%测试集输入
T_test = res(temp(201: end), 1)';%测试集输出
N = size(P_test, 2);

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);%训练集输入归一化到01之间
P_test = mapminmax('apply', P_test, ps_input);%测试集输入归一化到01之间

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺，神经网络工具箱要求4维数据，就是下面的7 1 1  M
P_train =  double(reshape(P_train, 4, 1, 1, M));%7 1 1  表示7*1*1的数据 M为训练集样本数目  
P_test  =  double(reshape(P_test , 4, 1, 1, N));

t_train = t_train';%输出进行转置，格式要求
t_test  = t_test' ;



%%  数据格式转换 转换成元胞数组的形式
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  创建模型
layers = [
    sequenceInputLayer(4)               % 建立输入层，根据几个特征写几个
    
    lstmLayer(4, 'OutputMode', 'last')  % LSTM层，4个隐藏层单元
    reluLayer                           % Relu激活层
    
    fullyConnectedLayer(1)              % 全连接层，就是输出层，1个输出就是1
    regressionLayer];                   % 回归层
 
%%  参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法，还有sgdm
    'MiniBatchSize', 30, ...               % 批量大小，每次拿多少样本训练，本案例中有80个数据，每次拿30个，训练两次，剩下的20个不要了。
    'MaxEpochs', 1200, ...                 % 最大迭代次数，训练1200论 1200*2=2400次
    'InitialLearnRate', 1e-2, ...          % 初始学习率为0.01
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.5, ...        % 学习率下降因子
    'LearnRateDropPeriod', 800, ...        % 经过 800 次训练后 学习率为 0.01 * 0.5
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);%关闭命令行显示的信息

%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);

%%  仿真预测
t_sim1 = predict(net, p_train);%训练集输入预测，看看模型训练如何
t_sim2 = predict(net, p_test );%测试集输入预测，评价模型训练如何

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%%  查看网络结构
analyzeNetwork(net)

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])

save('800 1200.mat','net')
disp('网络已保存')
save('800 1200para.mat','ps_input','ps_output')
disp('归一化参数已保存')