% group lasso function

% input
% X: explanatory variable
% y: response variable
% CV: the number of cross validation
% lambda_ratio: range of the regularization parameter to search
% lanbda_num: the number of the regularization parameter to search
% g: the number of the member in each group

% output
% beta_1SE: coefficient vector by 1SE rule
% beta_RMS: coefficient vector by min RMSE
% lambda: the regularization parametera (lambda(1)=1SE rule, lambda(2)=min RMSE)
% flag: convergence flag

function [beta_1SE,beta_RMS,lambda,flag] = grlasso(X,y,CV,lambda_ratio,lambda_num,g)

flag = 1;

% Data preprocessing
X_ext = zeros(size(X,1)*2,size(X,2)*2);
X_ext_temp = zeros(size(X,1)*2,size(X,2)*2);
for k = 1 : size(X,2)
    X_ext_temp(1:size(X,1),2*k-1) = X(:,k);
    X_ext_temp(size(X,1)+1:end,2*k) = X(:,k);
end
y_ext = zeros(size(y,1)*2,size(y,2));
for k = 1 : size(X,1)
    X_ext(2*k-1,:) = X_ext_temp(k,:);
    X_ext(2*k,:) = X_ext_temp(size(X,1)+k,:);
    y_ext(2*k-1,:) = real(y(k));
    y_ext(2*k,:) = imag(y(k));
end
clear X_ext_temp

[n,m] = size(X_ext);

% cumulative partition
cum_part = cumsum(g);
lambda_ini = 0;
beta_ini = zeros(m,1);

for conv = 1 : 10
    LAMBDA = linspace(lambda_ratio+lambda_ini,lambda_ini,lambda_num);
    LAMBDA = 10.^ LAMBDA;

    [beta,~] = gr_admm(X_ext,y_ext,LAMBDA(end),cum_part,beta_ini);
    if beta ~= 0
        lambda_ini = lambda_ini - lambda_ratio;
        beta_ini = beta;
    elseif beta == 0
        break
    end
end

if conv == 10
    flag = flag * 2;
end

for conv = 1 : 10
    LAMBDA = linspace(lambda_ratio+lambda_ini,lambda_ini,lambda_num);
    LAMBDA = 10.^ LAMBDA;

    [beta,~] = gr_admm(X_ext,y_ext,LAMBDA(1),cum_part,beta_ini);
    if beta == 0
        lambda_ini = lambda_ini + lambda_ratio;
        beta_ini = beta;
    elseif beta ~= 0
        break
    end
end

if conv == 10
    flag = flag * 3;
end

LAMBDA(1) = 0;

% check that sum(p) = total number of elements in x
if (sum(g) ~= m)
    error('invalid partition');
end

%% CV
% prepare for cross validation
MemberNum = zeros(CV,1);
for i = 1 : n
    MemberNum(mod(i-1,CV)+1) = MemberNum(mod(i-1,CV)+1) + 1;
end
CV_group = cumsum(MemberNum);
beta_ini = beta;

for l = length(LAMBDA) : -1 : 1
    lambda = LAMBDA(l);
    E_CV = zeros(size(CV_group));
    for j = 1 : length(CV_group)
        if j == 1
            sel = 1 : CV_group(j);
        else
            sel = CV_group(j-1)+1 : CV_group(j);
        end
        y_student = y_ext(sel);
        y_teacher = y_ext;
        y_teacher(sel) = [];
        X_student = X_ext(sel,:);
        X_teacher = X_ext;
        X_teacher(sel,:) = [];

        [beta,~] = gr_admm(X_teacher,y_teacher,lambda,cum_part,beta_ini);
        recon = X_student * beta;
        e = 0;
        for i = 1 : length(recon)
            e = e + (y_student(i)-recon(i))^2;
        end
        E_CV(j) = e / length(recon);
        beta_ini = beta;
    end
    E_ave(l) = mean(E_CV);
    E_std(l) = std(E_CV);
end

[E_min,min_index] = min(E_ave);

% 1SE
E_high = E_min + E_std(min_index);
index = find(E_ave<E_high,1,'last');
lambda_1SE = LAMBDA(index);
beta_ini = zeros(size(beta));
[beta_1SE,conv] = gr_admm(X_ext,y_ext,lambda_1SE,cum_part,beta_ini);
if conv == 0
    flag = flag * 5;
end

% min RMSE
lambda_RMS = LAMBDA(min_index);
beta_ini = beta_1SE;
[beta_RMS,conv] = gr_admm(X_ext,y_ext,lambda_RMS,cum_part,beta_ini);
if conv == 0
    flag = flag * 7;
end
lambda = [lambda_1SE lambda_RMS];

end