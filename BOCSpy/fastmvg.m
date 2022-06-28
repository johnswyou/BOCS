%% Fast sampler for multivariate Gaussian distributions (large p, p > n) of the form
%  N(mu, S), where
%       mu = S Phi' y
%       S  = inv(Phi'Phi + inv(D))
% Reference: 
%   Fast sampling with Gaussian scale-mixture priors in high-dimensional regression
%   A. Bhattacharya, A. Chakraborty and B. K. Mallick
%   arXiv:1506.04778

function x = fastmvg(Phi, ~, alpha, D)

[n,p] = size(Phi);

d = diag(D);
u = randn(p,1) .* sqrt(d);
delta = randn(n,1);
v = Phi*u + delta;
%w = (Phi*D*Phi' + eye(n)) \ (alpha - v);
%x = u + D*Phi'*w;
Dpt = bsxfun(@times, Phi', d);
w = (Phi*Dpt + eye(n)) \ (alpha - v);
x = u + Dpt*w;

end
