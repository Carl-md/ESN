clc;
N=2000;
K=4;
L=1;
n=10;
p=0.01;
%p为稀疏矩阵的连接率
%参数的选择参考了文献，中间层1000个，Win为(-1,1),Wback为(-0.1,0.1),W为幅值为0.8的稀疏矩阵。
%为了拟合结果比较合适，需要适当调节参数。以达到最优
Win=2*rand(N,K)-1;
Wback=0.1*(2*rand(N,L)-1);
W=0.8*sprandn(N,N,p);
U=[1 2 3 4 5 6 7 8 9 0
   2 3 4 5 6 7 8 9 0 1
   3 4 5 6 7 8 9 0 1 2
   4 5 6 7 8 9 0 1 2 3];
D=[1 2 3 4 5 6 7 8 9 10]';
X(N,n)=0;

X(:,1)=tanh(Win*U(:,1));
for i=1:1:n-1
   X(:,i+1)=tanh(Win*U(:,i+1)+W*X(:,i)+Wback*D(i));
end
M=X';
Wout=(pinv(M)*D)';
%得到了Wout后，即可以投入使用
m=12;
X1(N,m)=0;
Y(m)=0;
%U1可以任意定义

U1=[ 3 4 5 6 7 8 9 0 1 1 0 0 
    4 5 6 7 8 9 0 1 2 2  0 0
    5 6 7 8 9 0 1 2 3 3 0 0
    6 7 8 9 0 1 2 3 4 4 0 0];

X1(:,1)=tanh(Win*U1(:,1));
Y(1)=Wout*X1(:,1);
for i=1:1:m-1
    X1(:,i+1)=tanh(Win*U1(:,i+1)+W*X1(:,i)+Wback*Y(i));
   Y(i+1)=Wout*X1(:,i+1);
end
    Y(m)=Wout*X1(:,m);

    Y



