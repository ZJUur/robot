load('750 1250.mat');
load('750 1250para.mat');

% 1. 创建待预测数据并进行归一化（与训练时相同的步骤）
input_data_to_predict = [-1 1 3 -57]'; % 你要预测的四个数据
input_data_to_predict_normalized = mapminmax('apply', input_data_to_predict, ps_input); % 归一化输入数据

% 2. 转换为神经网络要求的格式（与训练时相同的步骤）
input_data_to_predict_reshape = double(reshape(input_data_to_predict_normalized, 4,1,1,1));

% 3. 使用神经网络进行预测
predicted_output_normalized = predict(net, input_data_to_predict_reshape);

% 4. 反归一化预测结果得到实际结果
predicted_output = mapminmax('reverse', predicted_output_normalized, ps_output);

% 打印预测结果
fprintf('预测结果为：%f\n', predicted_output);

