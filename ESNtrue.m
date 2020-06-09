%��������
data = Qtrain;
%���ô����ص���ز���
K = 3;%�����ڵ���
L = 1;%�����ڵ���
N = 100;%�����ؽڵ���
p = 0.1;%ϡ�����W�ڲ�������
n = 219;%������
rand( 'seed', 42 );%�̶������
Win = 0.9*(2*rand(N,K+L)-1);
W = 0.1*sprandn(N,N,p);
%���ž���
%��һ���������װ뾶
disp '�����װ뾶...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* (1.0/rhoW);
ans = eigs(W,1,'LM')
W = full(W);
%���ô����ص�״̬����
X = zeros(L+K+N,n);
%ֱ��������Ӧ��Ŀ�����
Yt = data(:,1)';
%�������ݣ�Ȼ��õ�ÿһʱ�̵�����ֵ�ʹ�����״̬
x = zeros(N,1);
for t = 1:n
	u = data(t,2:end)';
	x = tanh(Win*[1;u] + W*x);
    X(:,t) = [1;u;x];
end
%������ع�ѵ������
reg = 1e-1;%����ϵ��
%�ı��еķ���ʽΪ
X_T = X'; 
%Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
%����matlab���㣺
Wout = Yt*X_T/(X*X_T + reg*eye(L+K+N));
%run the trained ESN in a generative mode. no need to initialize here, 
%because x is initialized with training data and we continue from there.
Y = zeros(L,16);
for t = 1:16
    u = Gtest(t,2:end)';
	x = tanh( Win*[1;u] + W*x );
	y = Wout*[1;u;x];
	Y(:,t) = y;
	%����ģ��:
end
%������Լ���mse
Y = mapminmax('reverse',Y,testps);
G_test= mapminmax('reverse',Gtest',testps);
errorLen = 16;
mse = sum((G_test(1,1:16)-Y(1,1:errorLen)).^2)./errorLen;
disp( ['MSE = ', num2str( mse )] );
%����plotͼ����ʵ������Ԥ�����ݵ�plotֵ��
figure(1);
plot( G_test(1,1:16), 'color', [0,0.75,0] );
hold on;
plot( Y(1,1:16), 'b' );
hold off;
axis tight;
title('Ŀ��ֵ��Ԥ��ֵ');
legend('Ŀ��ֵ', 'Ԥ��ֵ');
ylabel('����ֵ/W')
xlabel('ʱ��/h')
mae=sum(abs(G_test(1,1:16)-Y(1,1:errorLen)))/errorLen;
disp( ['MAE = ', num2str( mae )] );
error = G_test(1,1:16)-Y(1,1:errorLen);
figure(2);
plot(error,'b');
hold off;
axis tight;
title('GRA-ESNԤ�����');
ylabel('���ֵ/W')
xlabel('ʱ��/h');