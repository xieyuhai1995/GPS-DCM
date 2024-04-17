function gp = GPmodel_BlackBox(x, y, l, u)
% function [gp, gp_classif] = GPmodel_BlackBox(x, y, l, u, x_classif, y_classif)
% Fits GP model for the objective function:
% normalised expected squared distance  used in Bayesian Optimization

% Hyperpriors

lik = lik_gaussian('sigma2', 0.1, 'sigma2_prior', prior_fixed);

alpha = 0.2;

lgt1 = (alpha * (u(1)-l(1))); % lengthscale for epsilon
lgt2 = (alpha * (u(2)-l(2))); % lengthscale for L

gpcf = gpcf_sexp('lengthScale', [lgt1, lgt2], 'magnSigma2', 1,...
    'lengthScale_prior', prior_fixed, 'magnSigma2_prior', prior_fixed);

jitter=0;%1e-9;

gp = gp_set('lik', lik, 'cf', gpcf,...
    'jitterSigma2', jitter);

end

