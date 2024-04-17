function [epsilon, nfevals] = find_reasonable_epsilon(theta0, grad0, logp0, f)

epsilon = 1;
r0 = randn(length(theta0), 1);
% Figure out what direction we should be moving epsilon.
[~, rprime, ~, logpprime] = leapfrog(theta0, r0, grad0, epsilon, f);
nfevals = 1;
acceptprob = exp(logpprime - logp0 - 0.5 * (rprime' * rprime - r0' * r0));
a = 2 * (acceptprob > 0.5) - 1;
% Keep moving epsilon in that direction until acceptprob crosses 0.5.
while (acceptprob^a > 2^(-a))
    epsilon = epsilon * 2^a;
    [~, rprime, ~, logpprime] = leapfrog(theta0, r0, grad0, epsilon, f);
    nfevals = nfevals + 1;
    acceptprob = exp(logpprime - logp0 - 0.5 * (rprime' * rprime - r0' * r0));
end

end

function [thetaprime, rprime, gradprime, logpprime] = leapfrog(theta, r, grad, epsilon, f)

rprime = r + 0.5 * epsilon * grad;
thetaprime = theta + epsilon * rprime;
[logpprime, gradprime] = f(thetaprime);
rprime = rprime + 0.5 * epsilon * gradprime;

end