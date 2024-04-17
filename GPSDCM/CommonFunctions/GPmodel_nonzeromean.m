function [gp_regr, nlml_regr] = GPmodel_nonzeromean(x_regr, y_regr, H_r, b, B)
% Fits and return GP regression to x and y

%% GP regression

% Hyperpriors
plg = prior_logunif();
pms = prior_logunif();
ps = prior_logunif();

for i = 1:size(H_r,1)
    lik = lik_gaussian('sigma2', H_r(end), 'sigma2_prior', ps);
    
    % We allow for different lengthscale in every dimension (ARD)
    % One magnitude as it is a 1 single output (rss 1x1)
    
    % use stationary (squared exponenetial) cov fct for 2nd & 3rd GP
    gpcf = gpcf_sexp('lengthScale', H_r(i,2:end-1), ...
        'magnSigma2', H_r(i,1),...
        'lengthScale_prior', plg, 'magnSigma2_prior', pms);
    
    % base functions for GP's mean function (2nd order polynomial).
    gpmf1 = gpmf_constant('prior_mean',b(1),'prior_cov',B(1,1));
    gpmf2 = gpmf_linear('prior_mean',b(2),'prior_cov',B(2,2));
    gpmf3 = gpmf_squared('prior_mean',b(5),'prior_cov',B(5,5),'interactions','on');
    
    % Set a small amount of jitter to be added to the diagonal elements of the
    % covariance matrix K to avoid singularities when this is inverted
    jitter=1e-8;
    
    % Create the GP structure
    gp_regr_all{i} = gp_set('lik', lik, 'cf', gpcf,...
        'meanf', {gpmf1,gpmf2,gpmf3},'jitterSigma2', jitter);
    
    %     % Sample
    %     [rgp_mcmc,g,opt]=gp_mc(gp_regr_all{i}, x_regr, y_regr, ...
    %     'nsamples', 300, 'display', 20);
    %     rr_regr = thin(rgp_mcmc,100,2);
    
    % Set the options for the optimization
    opt=optimset('TolFun',1e-6,'TolX',1e-6,'DerivativeCheck','on');
    
    % Optimize with the scaled conjugate gradient method
    [gp_regr_all{i}, nlml_regr(i)] = ...
        gp_optim(gp_regr_all{i},x_regr,y_regr,'opt',opt);
    
end

I = find(nlml_regr == min(nlml_regr));
gp_regr = gp_regr_all{I};
end

