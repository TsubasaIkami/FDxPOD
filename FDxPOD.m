clear all
close all
clc

%% data loading
load("X1.mat")
load("X2.mat")
load("X3.mat")
load("Xave.mat")
X = [X1 X2 X3]; % data matrix
clear X1 X2 X3

n = size(X,1); % the number of spatial points
m = size(X,2); % the number of time step
SR = 5000; % sampling rate [Hz]
p = 128; % the number of optimal sensors

%% SVD
[U,S,V] = svd(X,'econ'); % singular value decomposition
S = diag(S);

% truncation and data save
r = 16; % rank of POD mode to save
Ur = U(:,1:r);
Vr = V(:,1:r);
Sr = S(1:r);
clear U V S

%% optimal sensor placement
sensor = DGspsensor(Ur,p); % optimal sensor-placement by Saito et al.

num = zeros(size(adress)); % strage for numbering the spatial point
k = 1;
for i = 1 : length(adress)
    if adress(i) == 1 % if i is not in the mask
        num(i) = k; % number
        k = k + 1; % increment
    end
end

% pick up the data at the optimal sensor points
U_selected = zeros(p,r);
M_selected = zeros(p,m);
for i = 1 : p
    U_selected(i,:) = Ur(sensor(i),:); % POD mode at optimal sensor points
    M_selected(i,:) = X(sensor(i),:); % data at optimal sensor points
end
clear X
Phi_selected = fft(M_selected,[],2); % Fourier-transformed data at optimal sensor points

%% group lasso
% setting
lambda_ratio = -4; % range of the regularization parameter to search
lambda_num = 100; % the number of the regularization parameter to search
CV = 16; % the number of cross validation

% prepare for cross validation
MemberNum = zeros(CV,1);
for i = 1 : p
    MemberNum(mod(i-1,CV)+1) = MemberNum(mod(i-1,CV)+1) + 1; % make partition for CV
end
CV_group = cumsum(MemberNum); % group for the cross validation

half = ceil((m+1)/2); % below Nyquist frequency
B_1SE = zeros(r,m); % strage for amplitude Beta (1SE rule)
B_RMS = zeros(r,m); % strage for amplitude Beta (min RMSE)
lambda_1SE = zeros(half,1); % strage for the regularization parametera (1SE rule)
lambda_RMS = zeros(half,1); % strage for the regularization parameter (min RMSE)
flag_conv = zeros(half,1); % flag to check convergence
flag_group = ones(r,1)*2; % the number of the member in each group

parfor i = 1 : half
    disp(i)
    [beta_hat_1SE,beta_hat_MRS,lambda,flag] = grlasso(U_selected,Phi_selected(:,i),CV,lambda_ratio,lambda_num,flag_group); % call group lasso function
    beta_1SE = zeros(r,1); % strage for amplitude beta @ fi (1SE rule)
    beta_MRS = zeros(r,1); % strage for amplitude beta @ fi (min RMSE)
    for o = 1 : r
        beta_1SE(o) = beta_hat_1SE(2*o-1,:) + 1i*beta_hat_1SE(2*o,:); % odd index -> real part, even index -> imaginary part
        beta_MRS(o) = beta_hat_MRS(2*o-1,:) + 1i*beta_hat_MRS(2*o,:);
    end
    lambda_1SE(i,1) = lambda(1); % regularization parameter (1SE rule)
    lambda_RMS(i,1) = lambda(2); % regularization parameter (min RMSE)
    B_1SE(:,i) = beta_1SE; % store in B (1SE rule)
    B_RMS(:,i) = beta_MRS; % store in B (min RMSE)
    flag_conv(i) = flag; % convergence flag
end

% forming B considering the symmetricity of the Fourier-transform
B_half = conj(B_1SE(:,1:half));
B_half = fliplr(B_half);
if mod(half,2) == 0
    B_1SE(:,half+1:m) = B_half(:,2:half-1);
elseif mod(half,2) == 1
    B_1SE(:,half+1:m) = B_half(:,1:half-1);
end
B_half = conj(B_RMS(:,1:half));
B_half = fliplr(B_half);
if mod(half,2) == 0
    B_RMS(:,half+1:m) = B_half(:,2:half-1);
elseif mod(half,2) == 1
    B_RMS(:,half+1:m) = B_half(:,1:half-1);
end

%% reconstruction
A = ifft(B_1SE.'); % inv Fourier-transform to compute POD coefficients A. to use min RMSE result, change into B_RMS
M = Ur * A.'; % reconstruction
M = M + Xave; % add time-indipendent data (if necessary)

% arranging data
for t = 1 : size(B_1SE,2)
    temp = M(:,t);
    recon = zeros(size(adress));
    k = 1;
    for i = 1 : length(adress)
        if adress(i) == 1
            recon(i) = temp(k);
            k = k + 1;
        end
    end
    recon = reshape(recon,[128,128]);

    figure(1);
    imagesc(flipud(recon))
    clim([-15 15])
    axis equal
    axis off
    colormap jet
end
