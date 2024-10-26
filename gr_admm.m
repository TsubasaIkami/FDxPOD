% function of Alternating Direction Method of Multipliers:ADMM for group lasso

% input
% X: explanatory variable
% y: response variable
% lambda: regularization parameter
% cum_part: cumulative partition
% gamma: coefficient vector (inital value)

% output
% gamma: coefficient vector
% flag_conv: convergence flag

function [gamma,flag_conv] = gr_admm(X,y,lambda,cum_part,gamma)
% Global constants and defaults
MAX_ITER = 1000;
eps_abs = 1e-4;
eps_re = 1e-4;
tau_incr = 2;
tau_decr = 2;
flag_conv = 0;
[n,m] = size(X);
u = zeros(m,1);
rho = 1;
mu = 10;
M1 = X.'*X;
N = X.'*y;
M = inv(M1 + rho*eye(m,m));

for k = 1 : MAX_ITER

    gammaold = gamma;
    % beta-update
    beta = M * (N + rho*(gamma - u));
    % gamma-update
    start_ind = 1;
    for j = 1 : length(cum_part)
        sel = start_ind : cum_part(j);
        gamma(sel) = shrinkage(beta(sel) + u(sel), lambda/rho);
        start_ind = cum_part(j) + 1;
    end
    % u-update
    u = u + beta - gamma;

    % convergence
    r_norm  = norm(beta - gamma);
    s_norm  = norm(rho*(gamma - gammaold));
    eps_pri = sqrt(n)*eps_abs + eps_re*max([norm(beta) norm(gamma)]);
    eps_dual = sqrt(m)*eps_abs + eps_re*norm(rho*u);
    if r_norm <= eps_pri && s_norm <= eps_dual
        flag_conv = 1;
        break;
    end

    % rho-update
    if r_norm > mu*s_norm
        rho = tau_incr * rho;
        u = u / tau_incr;
        M = inv(M1 + rho*eye(m,m));
    elseif s_norm > mu*r_norm
        rho = rho / tau_decr;
        u = u * tau_decr;
        M = inv(M1 + rho*eye(m,m));
    end
end
