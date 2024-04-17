function gp = GPmodel_nESJD(x, y, l, u)
% Fits GP model for the objective function:
% normalised expected squared distance  used in Bayesian Optimization

% Hyperpriors
plg = prior_logunif();%prior_loggaussian('mu', 0.2, 's2', 0.5); % prior for lengthscale
pms = prior_logunif();%prior_sqrtunif(); % prior for magnSigma2

ps = prior_logunif(); % prior for sigma2 in the likelihood

lik = lik_gaussian('sigma2', 0.1, 'sigma2_prior', ps);

alpha = 0.2;

lgt1 = (alpha * (u(1)-l(1))); % lengthscale for epsilon
lgt2 = (alpha * (u(2)-l(2))); % lengthscale for L

gpcf = gpcf_sexp('lengthScale', [lgt1, lgt2], 'magnSigma2', 1,...
    'lengthScale_prior', plg, 'magnSigma2_prior', pms);

jitter=1e-9;

gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', jitter);

% Optimise hyperparameters
opt=optimset('TolFun',1e-6,'TolX',1e-6);

gp = gp_optim(gp,x,y,'opt',opt);

end

