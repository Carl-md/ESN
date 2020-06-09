%加载数据
data = Qtrain;
%设置储备池的相关参数
K = 3;%输入层节点数
L = 1;%输出层节点数
N = 100;%储备池节点数
p = 0.1;%稀疏矩阵W内部连接率
n = 219;%样本数
rand( 'seed', 42 );%固定随机数
Win = 0.9*(2*rand(N,K+L)-1);
W = 0.1*sprandn(N,N,p);
%缩放矩阵
%归一化并设置谱半径
disp '计算谱半径...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* (1.0/rhoW);
ans = eigs(W,1,'LM')
W = full(W);
%设置储备池的状态矩阵
X = zeros(L+K+N,n);
%直接设置相应的目标矩阵
Yt = data(:,1)';
%输入数据，然后得到每一时刻的输入值和储备池状态
x = zeros(N,1);
for t = 1:n
	u = data(t,2:end)';
	x = tanh(Win*[1;u] + W*x);
    X(:,t) = [1;u;x];
end
%运用岭回归训练产出
reg = 1e-1;%正则化系数
%文本中的方程式为
X_T = X'; 
%Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
%运用matlab解算：
Wout = Yt*X_T/(X*X_T + reg*eye(L+K+N));
%run the trained ESN in a generative mode. no need to initialize here, 
%because x is initialized with training data and we continue from there.
Y = zeros(L,16);
for t = 1:16
    u = Gtest(t,2:end)';
	x = tanh( Win*[1;u] + W*x );
	y = Wout*[1;u;x];
	Y(:,t) = y;
	%生成模型:
end
%计算测试集的mse
Y = mapminmax('reverse',Y,testps);
G_test= mapminmax('reverse',Gtest',testps);
errorLen = 16;
mse = sum((G_test(1,1:16)-Y(1,1:errorLen)).^2)./errorLen;
disp( ['MSE = ', num2str( mse )] );
%绘制plot图（真实数据与预测数据的plot值）
figure(1);
plot( G_test(1,1:16), 'color', [0,0.75,0] );
hold on;
plot( Y(1,1:16), 'b' );
hold off;
axis tight;
title('目标值与预测值');
legend('目标值', '预测值');
ylabel('功率值/W')
xlabel('时刻/h')
mae=sum(abs(G_test(1,1:16)-Y(1,1:errorLen)))/errorLen;
disp( ['MAE = ', num2str( mae )] );
error = G_test(1,1:16)-Y(1,1:errorLen);
figure(2);
plot(error,'b');
hold off;
axis tight;
title('GRA-ESN预测误差');
ylabel('误差值/W')
xlabel('时刻/h');