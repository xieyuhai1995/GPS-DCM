function [epsilon, Hbar, epsilonbar] = DualAveraging(i, delta, alpha_ave, ...
    davg_par)
% Adapts epsilon, the step size in the NUTS algorithm at every iteration
% using dual averaging

% Input:
% davg_par = [gamma, t0, kappa, mu, epsilonbar, Hbar] 
% come from the (i-1)^th DualAveraging call
gamma = davg_par(1); t0 = davg_par(2); kappa = davg_par(3);
mu = davg_par(4); epsilonbar = davg_par(5); Hbar = davg_par(6);

eta = 1 / (i + t0);
Hbar = (1 - eta) * Hbar + eta * (delta - alpha_ave);
epsilon = exp(mu - sqrt (i) / gamma * Hbar);
neta = i^(-kappa);
%if isreal(exp((1 - neta) * log(epsilonbar) + neta * log(epsilon)))
epsilonbar = exp((1 - neta) * log(epsilonbar) + neta * log(epsilon));
%else
%    disp('stop here')
%end
end

