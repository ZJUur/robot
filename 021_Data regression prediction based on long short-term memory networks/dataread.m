clear all;
close all;
clc

%对于类似的txt文件，不含有字符，只有数字
filename = 'C:\Users\yqh\Desktop\1.txt';
data=readmatrix(filename);
y=data(:,1);
x1=data(:,2);
x2=data(:,3);
x3=data(:,4);
x4=data(:,5);
plot(x2,y);

title('只改一根线x2');


clear all;
close all;
clc
filename2 = 'C:\Users\yqh\Desktop\outputdata.txt';
data2=readmatrix(filename2);
y=data2(:,1);
x1=data2(:,2);
x2=data2(:,3);
x3=data2(:,4);
x4=data2(:,5);
plot(x2,y);

title('更改两根线x2');
figure;

plot(x3,y);
title('更改两根线x3');


