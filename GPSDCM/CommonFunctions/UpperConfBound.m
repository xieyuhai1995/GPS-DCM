function [ucb, grad_ucb] = UpperConfBound(x_new, gp_E, x_E, y_E, ...
    mean_y_E, std_y_E, s, p, beta)
% function ucb = UpperConfBound(x_new, gp_E, x_E, y_E, mean_y_E, std_y_E, ...
%     gp_E_classif, x_E_classif, y_E_classif, s, p, beta)
% Evaluates the negative acquisition function (upper confidence bound) at x_new

% [E, Var] = gp_pred(gp_E, x_E, y_E, x_new);
% mu_s = (E * std_y_E + mean_y_E) * s;
% sigma = std_y_E * sqrt(Var);
% ucb = mu_s + p * sqrt(beta) * sigma;
% ucb = -ucb;
% 
% [~,C] = gp_trcov(gp_E, x_E);
% L = chol(C,'lower');
% a = L'\(L\y_E);
% 
% magnS2 = gp_E.cf{1}.magnSigma2;
% lgtScales = gp_E.cf{1}.lengthScale;
% Lambda = diag(1./(lgtScales.^2));
% 
% % Construct: magnS2 * exp(-0.5*((x-y)*Lambda*(x-y)'))
% AT = ( repmat(x_new,size(x_E,1),1) - x_E ) * Lambda;
% BT = ( repmat(x_new,size(x_E,1),1) - x_E )';
% 
% CT = sum(AT.*BT',2);
% sek = magnS2 .* exp(-0.5.*CT); sek = sek';
% 
% q1 = Lambda * (x_new - x_E)';
% q2 = sek' .* a;
% 
% grad_mu_s = - s * std_y_E * q1 * q2;
% 
% v = L\sek';
% 
% q5 = ( (v'/L) .* sek );
% 
% grad_sigma = std_y_E * 1/(2*sqrt(Var)) * ( 2*( q5 * q1' )' );
% 
% grad_ucb = grad_mu_s + p * sqrt(beta) * grad_sigma;
% 
% grad_ucb = -grad_ucb;

[E, Var] = gp_pred(gp_E, x_E, y_E, x_new);
mu_s = E * s;
sigma = sqrt(Var);
ucb = mu_s + p * sqrt(beta) * sigma;
ucb = -ucb;

[~,C] = gp_trcov(gp_E, x_E);
L = chol(C,'lower');
a = L'\(L\y_E);

magnS2 = gp_E.cf{1}.magnSigma2;
lgtScales = gp_E.cf{1}.lengthScale;
Lambda = diag(1./(lgtScales.^2));

% Construct: magnS2 * exp(-0.5*((x-y)*Lambda*(x-y)'))
AT = ( repmat(x_new,size(x_E,1),1) - x_E ) * Lambda;
BT = ( repmat(x_new,size(x_E,1),1) - x_E )';

CT = sum(AT.*BT',2);
sek = magnS2 .* exp(-0.5.*CT); sek = sek';

q1 = Lambda * (x_new - x_E)';
q2 = sek' .* a;

grad_mu_s = - s * q1 * q2;

v = L\sek';

q5 = ( (v'/L) .* sek );

grad_sigma = 1/(2*sqrt(Var)) * ( 2*( q5 * q1' )' );

grad_ucb = grad_mu_s + p * sqrt(beta) * grad_sigma;

grad_ucb = -grad_ucb;

end

