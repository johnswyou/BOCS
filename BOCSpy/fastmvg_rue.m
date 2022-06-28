%% Another sampler for multivariate Gaussians (small p) of the form
%  N(mu, S), where
%       mu = S Phi' y
%       S  = inv(Phi'Phi + inv(D))
%
% Here, PtP = Phi'*Phi (X'X is precomputed)
%
% Reference:
%   Rue, H. (2001). Fast sampling of gaussian markov random fields. Journal of the Royal
%   Statistical Society: Series B (Statistical Methodology) 63, 325–338.

function x = fastmvg_rue(Phi, PtP, alpha, D)

p = size(Phi,2);

Dinv = diag(1./diag(D));
%L = chol(Phi'*Phi + Dinv, 'lower');
% regularize PtP + Dinv matrix for small negative eigenvalues
try
    L = chol(PtP + Dinv, 'lower');
catch
    mat = PtP + Dinv;
    Smat = (mat + mat')/2;
    L = chol(Smat + max(eig(Smat))*(1e-15)*eye(size(Smat,1)),'lower');
end
v = L \ (Phi'*alpha);
m = L' \ v;
w = L' \ randn(p,1);

x = m + w;

end
