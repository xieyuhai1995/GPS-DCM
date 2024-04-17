function [LogPosterior, GradLogPosterior, Var, ObjFct] = ...
    HMCDerivPosterior_all_DCM(param_sc, ...
    sigma2, trueData, sigma, nd, ...
    em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
    gp_regr, x_regr, y_regr, mean_y, std_y, do_nuts, invLref, IS, Ep, M, U, V, row, column)


ns = size(y_regr,2); % no of species

invL = cell(ns,1);

for is=1:ns
    invL{is} = invLref(is).array; % inverse of the choleski decomp matrix for the emulator
    % covariance matrix passes over by reference
end

n = size(trueData,1);

if do_nuts == 1
    param_sc = param_sc';
end

Jacob = param_sc;

Sign = -1; % for rss to be used in phase_ind=1

if em_ind == 0
    
    if any(isnan(param_sc)) || any(~isfinite(param_sc)) || any(~isreal(param_sc))
        disp('The LogLikGradient is too high, change values of L and epsilon')
        ObjFct = repmat(10^10/ns,1,ns); pass = 0;
    else
        
        ObjFct = Run_simulator_DCM(param_sc, trueData, IS, Ep, M, U, V, row, column);
        
        if any(ObjFct) ~= 10^10/ns
            pass = 1;
        else
            pass = 0;
        end
    end

    func=@(param_sc) sum ( -n/2.*log(sigma2)-n/2*log(2*pi)*size(sigma2,1) - ObjFct./(2.*sigma2) );
    LogLik_tot = func(param_sc);
    
    if grad1_SimInd == 1 && pass == 1
        GradLogLik_tot = getGradient1(func,param_sc);
    else
        GradLogLik_tot = NaN(nd,1);
    end

    Var = Inf(1,ns);

else

    param_em = param_sc;

    ScJacob = Jacob';

    ObjFct = NaN(1,ns); E = NaN(1,ns); Var = NaN(1,ns); LogLik = NaN(ns,1);
    GradLogLik = NaN(nd,ns); 

    for is=1:ns
        
        if any(isnan(param_em)) || any(~isfinite(param_em)) || any(~isreal(param_em))
            disp('The LogLikGradient is too high, change values of L and epsilon')
            ObjFct(is) = 10^10/ns; pass = 0; Var(is) = Inf;

        else

            a = invL{is}' * (invL{is}*y_regr(:,is)); % (n,1)        % a = L'\(L\y_regr);
            magnS2 = gp_regr(is).cf{1}.magnSigma2; % (1,1)
            lgtScales = gp_regr(is).cf{1}.lengthScale; % (1,d)
            Lambda = diag(1./(lgtScales.^2)); % (d,d)

            % Construct: magnS2 * exp(-0.5*((x-y)*Lambda*(x-y)'))
            AT = ( repmat(param_em,size(x_regr,1),1) - x_regr ) * Lambda; % (n,d)
            BT = ( repmat(param_em,size(x_regr,1),1) - x_regr )'; % (d,n)

            CT = sum(AT.*BT',2)'; % (1,n)

            if all(isfinite(CT))
                sek = magnS2 .* exp(-0.5.*CT); % (1,n)
                
                E(is) = sek * a;
                v = invL{is}*sek';
                Var(is) = magnS2 - v'*v;
                
                if Var(is) < 1e-09
                    Var(is) = 1e-09;
                end
                
                pass = 1;
                
                if phase_ind == 1 % exploratory phase
                    ObjFct(is) = (E(is) + Sign * sqrt(Var(is))) * std_y(is) + mean_y(is); % E - sqrt(Var)
                else % phase_ind == 2 => sampling phase
                    ObjFct(is) = E(is) * std_y(is) + mean_y(is);
                end
                
            else
                
                ObjFct(is) = 10^10/ns; pass = 0; Var(is) = Inf;
                
            end
            
        end

        LogLik(is) = -n/2*log(sigma2(is))-n/2*log(2*pi) - ObjFct(is)/(2*sigma2(is));

        if pass == 0

            GradLogLik(:,is) = -(1/(2*sigma2(is))) * zeros(nd,1); % (d,1)

        else

            q1 = Lambda * BT; % (d,n)
            q4 = 1/(2*sigma2(is)) * std_y(is); % (1,1)

            FirstDerivKernel = - q1 .* sek; % (d,n) since (d,n) .* (1,n) = (d,n)

            FirstDerivGPpostMean = FirstDerivKernel * a; % first order derivative of the GP posterior mean (d,1)

            GradLogLik(:,is) = (-q4 * FirstDerivGPpostMean) .* ScJacob; % (d,1)

            if phase_ind == 1

                q5 = v'*invL{is};

                FirstDerivGPpostVar = -2 * q5 * FirstDerivKernel';
                FirstDerivGPpostVar = FirstDerivGPpostVar'; %(d,1)

                GradLogLik(:,is) = GradLogLik(:,is) + ( -q4 * Sign * 1/(2*sqrt(Var(is))) * ...
                    FirstDerivGPpostVar ) .* ScJacob; % (d,1)

            end % phase_ind

            if grad23_EmInd(1) == 1
                break;
            end

            if grad23_EmInd(2) == 1
                break;
            end

        end
    end
end

if em_ind == 1
    
    % sum likelihood terms 1 and 2 to get the total
    LogLik_tot = sum(LogLik); % (1,1)
    
    GradLogLik_tot = sum(GradLogLik,2); % (d,1)
    
end
count=ones(1,length(column));

for i=1:length(column)
    if row(i)==column(i)
        count(i)=1;
    else
        count(i)=0;
    end
end
res=length(param_sc)-length(column);
prior=-0*[count ones(1,res)];

LogPrior = sum(-0.5*log(2*pi)-log(sigma)-0.5*(((param_sc-prior)./sigma).^2));

GradLogPrior = - (param_sc-prior) ./ (sigma.^2);
GradLogPrior = GradLogPrior';

if pass == 1
    
    LogPosterior = LogLik_tot + LogPrior;
    
    GradLogPosterior = GradLogLik_tot + GradLogPrior;

else % pass = 0
    
    LogPosterior = -10^10; % equivalent to Posterior = 0
    GradLogPosterior = zeros(nd,1);

end

end



