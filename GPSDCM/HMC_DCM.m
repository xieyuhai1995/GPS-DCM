function [current_p,LogPosterior_sim,LogPosterior_em,GradLogPost_em,...
    ObjFct_sim, noODE_counter, varargout] = ...
    HMC_DCM(current_p, sigma2, epsilon, L, ...
    gp_regr, x_regr, y_regr, nd, phase_ind, trueData, ...
    sigma, ...
    ObjFct_sim_begin, mean_y, std_y, do_nuts, Moment, invLref, do_DA, IS, Ep, M, U, V, row, column)

noODE_counter = 0; % count the no of ODE evaluations

p = current_p;

% Update the momentum variable, q ~ MVN(0,I)
q = mvnrnd(zeros(nd,1),eye(nd));
current_q = q;

% re-obtain LogPosterior_em_begin,GradLogPost_em_begin and
% LogPosterior_sim_begin if changing sigma2 from iteration to iteration
% and re-obtain LogPosterior_em_begin,GradLogPost_em_begin only in
% exploratory phase when we re-fit GP from iteration to iteration, as the
% predicted value will change

em_ind = 1; % use emulator
grad1_SimInd = NaN; grad23_EmInd = [0 0];

[LogPosterior_em_begin,GradLogPost_em_begin] = ...
    HMCDerivPosterior_all_DCM(p, ...
    sigma2, trueData, sigma, nd, ...
    em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
    gp_regr, x_regr, y_regr, mean_y, std_y, do_nuts, invLref,IS, Ep, M, U, V, row, column);


n = size(trueData,1);
count=ones(1,length(column));

for i=1:length(column)
    if row(i)==column(i)
        count(i)=1;
    else
        count(i)=0;
    end
end
res=length(p)-length(column);
prior=-0*[count ones(1,res)];
LogPosterior_sim_begin = sum ( -n/2.*log(sigma2)-n/2*log(2*pi)*size(sigma2,1) - ...
        ObjFct_sim_begin./(2.*sigma2)) + sum(-0.5*log(2*pi)-log(sigma)-0.5*(((p-prior)./sigma).^2));


current_E_kin = 0.5 * current_q/Moment*current_q';

% Use the gradient of the log posterior density of p to make half step of q
q = q + 0.5 * epsilon * GradLogPost_em_begin';

for i = 1:L % L: no of steps
    % Make a full step for the position
    p = p + epsilon * (Moment\q')';
    
    % Make a full step for the momentum, except at end of trajectory
    if i~=L
        em_ind = 1;%0; % use emulator within the trajectory
        grad1_SimInd = NaN;
        grad23_EmInd = [0 0];
        %
        %[~, GradLogPost_trj, Var, ~] = HMCDerivPosterior_all_DCM(p, ...
        %    sigma2, trueData, sigma, nd,  ...
        %    em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
        %    gp_regr, x_regr, y_regr, mean_y, std_y, do_nuts, invLref, IS, Ep, M, U, V, row, column);
        [~,GradLogPost_em_begin,Var,~] = ...
            HMCDerivPosterior_all_DCM(p, ...
            sigma2, trueData, sigma, nd, ...
            em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
            gp_regr, x_regr, y_regr, mean_y, std_y, do_nuts, invLref,IS, Ep, M, U, V, row, column);
        if phase_ind == 1 % exploratory
            % if sqrt(Var) >= 3, stop simulation and evaluate expensive target
            % density and its gradient at the point where we stopped
            
            if any(sqrt(Var) >= 3)
                disp('var too high')
                break
            end
        end
        
        %q = q + epsilon * GradLogPost_trj';
        q = q + epsilon * GradLogPost_em_begin';

        
    end
end

if phase_ind == 1 && any(sqrt(Var)>=3) &&( ~isempty(p) && isreal(p) && all(isfinite(p)) && all(~isnan(p)) )
    
    ObjFct = Run_simulator_DCM(p, trueData, IS, Ep, M, U, V, row, column);
    
    noODE_counter = noODE_counter + 1;
    
    x_regr(end+1,:) = p;

    y_regr = y_regr .* std_y + mean_y; % bring on original scale
    y_regr(end+1,:) = ObjFct; % original scale
    
    mean_y = mean(y_regr);
    std_y = std(y_regr);
    y_regr = (y_regr - mean_y)./std_y; 

    ns = size(trueData,2);

    for is=1:ns
        gp_regr(is) = gp_optim(gp_regr(is),x_regr,y_regr(:,is));
        
        % Update cov matrix here to pass on by reference
        [~,Cov] = gp_trcov(gp_regr(is), x_regr);
        L = chol(Cov,'lower');
        invL = L\eye(size(y_regr,1),size(y_regr,1));
        invLref(is).array = invL;
    end
                    
end 

if phase_ind == 1 % the script calling this function will not know
    % if the GP update has been done, so it'll expect these arguments in phase_ind = 1
    varargout{1} = gp_regr; varargout{2} = x_regr; varargout{3} = y_regr;
    varargout{4} = mean_y; varargout{5} =  std_y; varargout{6} = invLref;
else
    varargout = {};
end

% Make a half step for momentum at the end
em_ind = 1; % use emulator
grad1_SimInd = NaN; grad23_EmInd = [0 0];
[LogPosterior_em_end,GradLogPost_em_end] = HMCDerivPosterior_all_DCM(p, ...
    sigma2, trueData, sigma, nd, ...
    em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
    gp_regr, x_regr, y_regr, mean_y, std_y, do_nuts, invLref, IS, Ep, M, U, V, row, column);

q = q + 0.5 * epsilon * GradLogPost_em_end';

% Negate momentum at end of trajectory to make the proposal symmetric
q = -q;

% Evaluate potential and kinetic energy at end of trajectory
% H = E_pot + E_kin; E_pot = -LogPost; E_kin = 0.5*sum(q.^2);
% joint posterior distr: p(p,q) = exp(-H)
proposed_E_kin = 0.5 * q/Moment*q';

if do_DA == 0 % no delayed acceptance
    
    % Accept or reject the state at end of trajectory
    % (in a Metropolis step, with proposal distribution coming from the hamiltonian dynamics on emulated space,
    % returning either the position p at end of trajectory or the initial position
    
    % current-proposed and not proposed-current as it would normally because
    % because proposed is actually -logpost
    
    em_ind = 0; % use simulator at end of trajectory
    grad1_SimInd = 0; grad23_EmInd = [NaN NaN];
    [LogPosterior_sim_end, ~, ~, ObjFct_sim_end] = ...
        HMCDerivPosterior_all_DCM(p, ...
        sigma2, trueData, sigma, nd,  ...
        em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
        gp_regr, x_regr, y_regr, mean_y, std_y, do_nuts, invLref, IS, Ep, M, U, V, row, column);
    
    noODE_counter = noODE_counter + 1;
    
    r = - LogPosterior_sim_begin + LogPosterior_sim_end + current_E_kin - proposed_E_kin;
    
    %[current_E_pot, proposed_E_pot]
    %[current_E_kin, proposed_E_kin]
    
    if r > 0 || (r > log(rand)) % accept
        disp('accept')
        LogPosterior_sim = LogPosterior_sim_end;
        LogPosterior_em = LogPosterior_em_end;
        GradLogPost_em = GradLogPost_em_end;
        ObjFct_sim = ObjFct_sim_end;
        current_p = p;
        
    else % reject
        disp('reject')
        LogPosterior_sim = LogPosterior_sim_begin;
        LogPosterior_em = LogPosterior_em_begin;
        GradLogPost_em = GradLogPost_em_begin;
        ObjFct_sim = ObjFct_sim_begin;
        current_p = current_p;
        
    end
end