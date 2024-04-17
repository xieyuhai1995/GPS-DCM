function gp_regr = ReFitGP(x_regr, y_regr)
% Fits and return GP regression  model to x and y

%% GP regression

% Hyperpriors

plg = prior_logunif();
pms = prior_logunif();
ps = prior_logunif();

% %Sample
% lik = lik_gaussian('sigma2', 1.1, 'sigma2_prior', ps);
% gpcf = gpcf_sexp('lengthScale', [1 1 1 1], 'magnSigma2', 10^6, ...
% 'lengthScale_prior', plg, 'magnSigma2_prior', pms);
% gp_regr = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-4);
% [rgp_mcmc,g,opt]=gp_mc(gp_regr, x_regr, y_regr, 'nsamples', 400, 'display', 20);
% gp_regr = thin(rgp_mcmc,100,2);


% We allow for different lengthscale in every dimension (ARD)
% One magnitude as it is a 1 single output (rss 1x1)
lik = lik_gaussian('sigma2', exp(-9.950929526405522), 'sigma2_prior', ps);

gpcf = gpcf_sexp('lengthScale', 0.01*ones(1,size(x_regr,2)), ...
    'magnSigma2', 1,...
    'lengthScale_prior', plg, 'magnSigma2_prior', pms);

% gpcf = gpcf_neuralnetwork('biasSigma2', 1, ...
%             'weightSigma2', 100*ones(1,size(x_regr,2)), ...
%             'biasSigma2_prior',prior_logunif(), ...
%             'weightSigma2_prior',prior_logunif());
        
% Set a small amount of jitter to be added to the diagonal elements of the
% covariance matrix K to avoid singularities when this is inverted
jitter=1e-9;

% Create the GP structure
gp_regr = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', jitter);

% Set the options for the optimization
opt=optimset('TolFun',1e-6,'TolX',1e-6);

% Optimize with the scaled conjugate gradient method
gp_regr = gp_optim(gp_regr,x_regr,y_regr,'opt',opt);

% % Sample
% rgp_mcmc = gp_mc(gp_regr, x_regr, y_regr, 'nsamples', 300, 'display', 20);
% gp_regr = thin(rgp_mcmc,100,2);

end
