function [gp_regr, nlml_regr] = GPmodel(x_regr, y_regr, H_r)
% Fits and return GP regression to x and y

%% GP regression

% Hyperpriors
plg = prior_logunif();
pms = prior_logunif();
ps = prior_logunif();


% %Sample
% lik = lik_gaussian('sigma2', H_r(end), 'sigma2_prior', ps);
% gpcf = gpcf_sexp('lengthScale', H_r(2:end-1), 'magnSigma2', H_r(1), ...
% 'lengthScale_prior', plg, 'magnSigma2_prior', pms);
% gp_regr = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-4);
% [rgp_mcmc,g,opt]=gp_mc(gp_regr, x_regr, y_regr, 'nsamples', 1000, 'display', 20);
% rr_regr = thin(rgp_mcmc,500,2);
% gp_regr = rr_regr; nlml_regr=100;

for i = 1:size(H_r,1)
    lik = lik_gaussian('sigma2', H_r(end), 'sigma2_prior', ps);
    %lik = lik_t('sigma2', H_r(i,6), 'sigma2_prior', ps);
    
    % We allow for different lengthscale in every dimension (ARD)
    % One magnitude as it is a 1 single output (rss 1x1)
    
    % use stationary (squared exponenetial) cov fct for 2nd & 3rd GP
    gpcf = gpcf_sexp('lengthScale', H_r(i,2:end-1), ...
        'magnSigma2', H_r(i,1),...
        'lengthScale_prior', plg, 'magnSigma2_prior', pms);
    
    
    % Set a small amount of jitter to be added to the diagonal elements of the
    % covariance matrix K to avoid singularities when this is inverted
    jitter=1e-6;
    
    % Create the GP structure
    gp_regr_all{i} = gp_set('lik', lik, 'cf', gpcf,...
        'jitterSigma2', jitter);%, 'latent_method', 'EP');
    
    %     % Sample
    %     [rgp_mcmc,g,opt]=gp_mc(gp_regr_all{i}, x_regr, y_regr, ...
    %     'nsamples', 300, 'display', 20);
    %     rr_regr = thin(rgp_mcmc,100,2);
    
    % Set the options for the optimization
    opt=optimset('TolFun',1e-2,'TolX',1e-2);
    
    % Optimize with the scaled conjugate gradient method
    [gp_regr_all{i}, nlml_regr(i)] = ...
        gp_optim(gp_regr_all{i},x_regr,y_regr,'opt',opt);
    disp(strcat('GPmodel ',num2str(i)));
end

I = find(nlml_regr == min(nlml_regr));
gp_regr = gp_regr_all{I};
end

