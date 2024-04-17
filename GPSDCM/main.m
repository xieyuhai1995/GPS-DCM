close all;
clc;
clear all;

load('data7nodes_TR2_SNR10.mat')

repeat = 1;
T = 300;
n = 7;
TR = 1;
SNR = 10;

% True parameters
U.u=spm_rand_mar(T,n,1/2)/4;
U.dt=TR;
P.A=At;
[r,c]=size(P.A);
P.B=zeros(r,c);
P.C=eye(r,c);
P.D=zeros(r,c);
P.decay=0.1;
P.transit=[0.2,0.1,0.3,-0.2,0.4,-0.15,0.5]';
% P.transit=[0.2,-0.2,-0.15]';
P.epsilon=0.5;
M.x=zeros(r,5);
M.f=@spm_fx_fmri;
M.g=@spm_gx_fmri;
%M.dt=0.5;

y=spm_int_J(P,M,U);
y_gt=y;

Rn=eye(300);
alpha=0.9;
for i=1:300
    for j=1:300
        Rn(i,j)=alpha^(abs(i-j));
    end
end
R=chol(Rn);

for i=1:r
    sigma=std(y(:,i))*10^(-SNR/10);
    y(:,i)=y(:,i)+sigma*R'*randn(300,1);
end
e_gaussian=y-y_gt;

options.nonlinear  = 0;
options.two_state  = 0;
options.stochastic = 1;
options.centre     = 1;
options.induced    = 1;

DCM.options=options;
DCM.a=logical(P.A);
DCM.b=zeros(n,n,0);
DCM.c=zeros(n,1);
DCM.d=zeros(n,n,0);
DCM.Y.y=y;
DCM.U.u=zeros(300,1);

DCM.U.dt=TR;
DCM.Y.dt=TR;

DCM1 = spm_dcm_fmri_csd(DCM);

[y_csd,IS,Ep,M,U,V] = spm_dcm_fmri_csd_MCMC(DCM);
y_csd = spm_vec(y_csd);
trueData = y_csd;
span=size(trueData,1);

[row,column]=find(DCM1.Ep.A~=0);
size_row=size(row,1);

% Generate the data
ns = 1; % no of species
nd = size_row + 1 + n + 1 + 2 + 2 + n; % total no of parameters without the error variance

% noise variance (same for each species)


% compact support for the emulator
l = -1*ones(1,nd);
u = ones(1,nd);

% Parameter scaling for emulation

X = sobolset(nd, 'Skip',1.4e4,'Leap',0.01e13); % 45000
%X = sobolset(nd, 'Skip',1.4e4,'Leap',2e14);

par = NaN(size(X,1),nd);
for i=1:nd
    par(:,i) = l(i) + (u(i)-l(i)) * X(:,i);
end

noODE_counter_InitExpl = 0; % count no of ODE evaluations in the initial & exploratory phase

delete(gcp('nocreate'))
parpool('local', 30)
% Run simulator to obtain rss for training points
parfor i=1:size(X,1)
    rss(i,1) = Run_simulator_DCM(par(i,:), y_csd, IS, Ep, M, U, V, row, column);
end

delete(gcp)

noODE_counter_InitExpl = noODE_counter_InitExpl + size(X,1);

k = 5000; % no of points we want to keep to fit the GP regression

temp = sort(sum(rss,2), 'ascend');
T_ss = temp(k);

I_ss = sum(rss,2) < T_ss;
x_regr = par(I_ss,:); 
y_regr = rss(I_ss,:);

mean_y = mean(y_regr);
std_y = std(y_regr);

y_regr = (y_regr-mean_y)./std_y;

X_r = sobolset(nd+2, 'Skip',2e12,'Leap',0.45e15); % draw 60 values
n_r = size(X_r, 1);

l_r = 0.1*ones(1,43);
u_r = 1*ones(1,43);
l_r(1)= 1;
u_r(1)= 10;
l_r(end) = 0.1;
u_r(end) = 0.01;


% H_r matrix with magnsigma2, every lengthscale and sigma2 on separate rows
for i=1:nd+2
    H_r(:,i)= l_r(i)+(u_r(i)-l_r(i)).*X_r(:,i);
end

gp_regr = GPmodel(x_regr, y_regr, H_r);

[w,s] = gp_pak(gp_regr);
disp(exp(w))

%Make predictions using gp_regr
[E, Var] = gp_pred(gp_regr, x_regr, y_regr, x_regr);
figure; clf; plot(y_regr, E, '.');
hold on; plot(y_regr,y_regr,'-r')
xlabel('Train data'); ylabel('Predictions')

phase_ind = 1; % phase 1 in GPHMC algorithm

sigma = 0.05*ones(1,41);

% for the noise variance (IG)
a = 0.001; b = 0.001; % uninformative

x_regr_refitted = x_regr;
y_regr_refitted = y_regr;

gp_regr_refitted = gp_regr;

% Obtain the covariance matrix and find its inverse to pass on by reference
% to the functions as to avoid the repeated matrix inversion
[~,Cov] = gp_trcov(gp_regr_refitted, x_regr_refitted);
Lcov = chol(Cov,'lower');
invL = Lcov\eye(size(y_regr_refitted,1),size(y_regr_refitted,1)); % inverse of Lcov
% Cov = Lcov'Lcov; Cov^(-1) = (Lcov'Lcov)^(-1) = (Lcov')^(-1)(Lcov^(-1)) = (Lcov^(-1))'(Lcov^(-1))
% (Lcov')^(-1) = (Lcov^(-1))', inverse of transpose is transpose of inverse for a triangular matrix
invLref = largematrix; % create an object to pass the inverse of Lcov as an argument by reference
invLref.array = invL;

do_nuts = 0;

nSamples = 15000;

p = NaN(nSamples,nd); % theta parameter samples

sigma2 = 0.2;

% Initialise
p(1,:) = x_regr(1,:);% this is unbounded variable

em_ind = 0; grad1_SimInd = 0; grad23_EmInd = [NaN NaN];

[LogPosterior_sim, ~, ~, ss(1,:)] = ...
    HMCDerivPosterior_all_DCM(p(1,:), sigma2, trueData, ...
    sigma, nd, em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
    gp_regr_refitted, x_regr_refitted, ...
    y_regr_refitted, mean_y, std_y, do_nuts, invLref, IS, Ep, M, U, V, row, column);

noODE_counter_InitExpl = noODE_counter_InitExpl + size(X,1);

em_ind = 1; grad1_SimInd = NaN; grad23_EmInd = [0 0];

[LogPosterior_em, GradLogPost_em, ~, ~] = ...
    HMCDerivPosterior_all_DCM(p(1,:), sigma2, trueData, ...
    sigma, nd, em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
    gp_regr_refitted, x_regr_refitted, ...
    y_regr_refitted, mean_y, std_y, do_nuts, invLref, IS, Ep, M, U, V, row, column);


acc = 0; % acceptance rate

Moment = 0.01*eye(nd,nd);

do_DA = 0; % do DA in the exploratory phase
% (it doesn't matter what we use in the exploratory phase when we construct the emulator)

% in exploratory phase, draw eps and L from uniform with the
% following bounds:
%l_eps = 0.0005; u_eps = 0.005;
epsilon = 0.001; L = 6;

j = 1;

T_ss1 = quantile(y_regr_refitted(:,1).*std_y(1)+mean_y(1),0.5);

T_ss = T_ss1;

opt=optimset('TolFun',1e-2,'TolX',1e-2);

rng('shuffle')

for i=2:nSamples
    disp(strcat('explore ',num2str(i-1),'sample'))
    [p(i,:),LogPosterior_sim,LogPosterior_em,GradLogPost_em, ss(i,:), ...
        noODE_counter_iter, gp_regr_refitted, ...
        x_regr_refitted, y_regr_refitted, mean_y, std_y, invLref] = ...
        HMC_DCM(p(i-1,:), sigma2, epsilon, L, ...
        gp_regr_refitted, x_regr_refitted, y_regr_refitted, ...
        nd, phase_ind, trueData, sigma, ...
        ss(i-1,:), mean_y, std_y, do_nuts, Moment, ...
        invLref, do_DA, IS, Ep, M, U, V, row, column);

    noODE_counter_InitExpl = noODE_counter_InitExpl + noODE_counter_iter;

    if all(p(i,:) ~= p(i-1,:)) % we've just accepted the new point

        %sum(ss(i,:))

        acc = acc + 1;

        % starting from beginning, gradually remove the old [size(y_regr,1)] training
        % points whose rss > T_ss, as we accept new train points & refit GP
        if acc <= size(y_regr,1) % delete or skip when we've accepted
            y_regr_refitted = y_regr_refitted .* std_y + mean_y;
            x_regr_refitted(j,:) = []; y_regr_refitted(j,:) = [];
            mean_y = mean(y_regr_refitted); std_y = std(y_regr_refitted);
            y_regr_refitted = (y_regr_refitted - mean_y)./std_y;

        else
            j = j + 1; % skip deleting
        end
    end

    if x_regr_refitted(end,:)~=p(i,:) % i.e. if we haven't already
        % added this point as a consequence of sqrt(Var)>=3

        x_regr_refitted(end+1,:) = p(i,:);
        y_regr_refitted = y_regr_refitted .* std_y + mean_y; % bring on original scale
        y_regr_refitted(end+1,:) = ss(i,:); % original scale

        mean_y = mean(y_regr_refitted);
        std_y = std(y_regr_refitted);
        y_regr_refitted = (y_regr_refitted - mean_y)./std_y;

        for is=1:ns
            gp_regr_refitted(is) = gp_optim(gp_regr_refitted(is),...
                x_regr_refitted,y_regr_refitted(:,is),'opt',opt);

            % Update cov matrix here to pass on by reference
            [~,Cov] = gp_trcov(gp_regr_refitted(is), x_regr_refitted);
            Lcov = chol(Cov,'lower');
            invL = Lcov\eye(size(y_regr_refitted,1),size(y_regr_refitted,1)); % inverse of Lcov
            % Cov = Lcov'Lcov; Cov^(-1) = (Lcov'Lcov)^(-1) = (Lcov')^(-1)(Lcov^(-1)) = (Lcov^(-1))'(Lcov^(-1))
            % (Lcov')^(-1) = (Lcov^(-1))', inverse of transpose is transpose of inverse for a triangular matrix
            %invLref = largematrix; % create an object to pass the inverse of Lcov as an argument by reference
            invLref(is).array = invL;
        end

    end

end

if i>nSamples/2
    sigma2 = 1./gamrnd(a+0.5*span, 1./(b+0.5*ss(i,:)));
end

epsilon = 0.001*rand; L = 10*rand;

y_regr_refitted = y_regr_refitted .* std_y + mean_y;

k = 500; % no of points we want to keep to fit the GP regression
temp = sort(sum(y_regr_refitted,2), 'ascend');
T_ss = temp(k);
I_ss = find(sum(y_regr_refitted,2) < T_ss);

y_regr_refitted = y_regr_refitted(I_ss,:);
x_regr_refitted = x_regr_refitted(I_ss,:);
mean_y = mean(y_regr_refitted);
std_y = std(y_regr_refitted);

y_regr_refitted = (y_regr_refitted - mean_y)./std_y;

% Refit GP with burnin phase removed
% Use this GP in the sampling phase
for is=1:ns
    gp_regr_refitted(is) = gp_optim(gp_regr_refitted(is),...
        x_regr_refitted,y_regr_refitted(:,is),'opt',opt);

    % Update the covariance matrix and its inverse
    [~,Cov] = gp_trcov(gp_regr_refitted(is), x_regr_refitted);
    Lcov = chol(Cov,'lower');
    invL = Lcov\eye(size(y_regr_refitted,1),size(y_regr_refitted,1)); % inverse of Lcov
    % Cov = Lcov'Lcov; Cov^(-1) = (Lcov'Lcov)^(-1) = (Lcov')^(-1)(Lcov^(-1)) = (Lcov^(-1))'(Lcov^(-1))
    % (Lcov')^(-1) = (Lcov^(-1))', inverse of transpose is transpose of inverse for a triangular matrix
    %invLref = largematrix; % create an object to pass the inverse of Lcov as an argument by reference
    invLref(is).array = invL;
end

do_DA = 0; % do delayed acceptance

hd = 2; % no of hyperparameters for HMC (epsilon, L)
% Set lower and upper bounds for epsilon and L in HMC to be used in
% Bayesian optimization
l_HMC = [5*10^(-4), 1];
u_HMC = [10^(-2), 20];

Moment = 0.01*eye(nd,nd);

X = sobolset(hd, 'Skip',1.4e4,'Leap',0.45e15); % draw 21 points
np = size(X, 1);

% Use first 20 points from Sobol sequence to build initial GP model
epsilon_init = l_HMC(1) + (u_HMC(1)-l_HMC(1)) * X(1:np,1);
L_init = round(l_HMC(2) + (u_HMC(2)-l_HMC(2)) * X(1:np,2)); % L integer

% Set number of HMC samples to generate for every (eps, L) pair
nSamples = 11; % draw nSamples-1 for every (eps,L) starting from previous run sample
% (that'll be the 1st out of nSamples)

% Sampling phase in Rasmussen's paper
phase_ind = 2;

% Pre-allocate memory for chains
p_sample_init = NaN((nSamples-1)*(np-1)+1, nd); % parameter samples from initial stage
nESJD_init = NaN(np,1); % normalised ESJD from initial stage (only the successful runs (nESJD>0))

ss_sample_init = NaN((nSamples-1)*(np-1)+1, ns); % sum-of-square samples from the sampling phase
s2_sample_init = NaN((nSamples-1)*(np-1)+1, ns); % sigma2 samples from the sampling phase

% Initialise
i = round(500*rand);
p_sample_init(1,:) = x_regr_refitted(i,:);% this is unbounded variable
s2_sample_init(1,:) = 0.05;

noODE_counter_preproc = 0; % count no of ODE evaluations in the pre-processing phase

em_ind = 0; grad1_SimInd = 0; grad23_EmInd = [NaN NaN];
[LogPosterior_sim, ~, ~, ss_sample_init(1,:)] = ...
    HMCDerivPosterior_all_DCM(p_sample_init(1,:), ...
    s2_sample_init(1,:), trueData, sigma, nd, ...
    em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
    gp_regr_refitted,x_regr_refitted, ...
    y_regr_refitted, mean_y, std_y, do_nuts, invLref, IS, Ep, M, U, V, row, column);

noODE_counter_preproc = noODE_counter_preproc + 1;

em_ind = 1; grad1_SimInd = NaN; grad23_EmInd = [0 0];
[LogPosterior_em,GradLogPost_em, ~, ~] = ...
    HMCDerivPosterior_all_DCM(p_sample_init(1,:), ...
    s2_sample_init(1,:), trueData, ...
    sigma, nd, em_ind, phase_ind, grad1_SimInd, ...
    grad23_EmInd, gp_regr_refitted, ...
    x_regr_refitted, y_regr_refitted, mean_y, std_y, do_nuts, invLref, IS, Ep, M, U, V, row, column);

acc = 0; % acceptance rate
next = 1;

initime = cputime();

tic()

rng('shuffle')

for j = 1:np-1 % iterate through different (eps, L) pairs
    % to obtain nSamples for each pair
    for i=2:nSamples

        next = next + 1;
        [p_sample_init(next,:),LogPosterior_sim,LogPosterior_em,...
            GradLogPost_em, ss_sample_init(next,:), noODE_counter_iter] = ...
            HMC_DCM(p_sample_init(next-1,:), ...
            s2_sample_init(next-1,:), epsilon_init(j), L_init(j), ...
            gp_regr_refitted, x_regr_refitted, y_regr_refitted, ...
            nd, phase_ind, trueData, sigma, ...
            ss_sample_init(next-1,:), ...
            mean_y, std_y, do_nuts, Moment, invLref, do_DA ,IS, Ep, M, U, V, row, column);

        noODE_counter_preproc = noODE_counter_preproc + noODE_counter_iter;

        if all(p_sample_init(next,:) ~= p_sample_init(next-1,:)) % we've just accepted the new point
            disp('accept')
            acc = acc + 1;
        else
            disp('reject')
        end

        s2_sample_init(next,:) = 1./gamrnd(a+0.5*span, 1./(b+0.5*ss_sample_init(next,:)));

    end

    xj = p_sample_init(next-nSamples+1:next,:);
    nESJD_init(j) = ESJDfct(xj)/sqrt(L_init(j));
    %nESJD_init(j)

end

x_E = [epsilon_init(1:np-1), L_init(1:np-1)]; % (np,2) size
y_E = nESJD_init(1:np-1);

gp_E = GPmodel_nESJD(x_E, y_E, l_HMC, u_HMC);

[w,~] = gp_pak(gp_E);
disp(exp(w))

%Make predictions using gp_E
[E, Var] = gp_pred(gp_E, x_E, y_E, x_E);
figure(1); clf(1); plot(y_E, E, '.', 'markersize', 20);
hold on; plot(y_E,y_E,'-r')
xlabel('Train data'); ylabel('Predictions')


%explore best epsilon,L
delta = 0.1;

% Set maximum no of BO iterations
% no of HMC samples to generate for every (eps, L) pair is 1 + (nSamples-1)*maxiter
maxiter = 20;

epsilon_adapt = NaN(maxiter,1); % stores the optimum step size from every BO iteration
L_adapt = NaN(maxiter,1); % stores the optimum no of leapfrog steps from every BO iteration

epsilon_adapt(1) = l_HMC(1) + (u_HMC(1)-l_HMC(1)) * X(np,1);
L_adapt(1) = round(l_HMC(2) + (u_HMC(2)-l_HMC(2)) * X(np,2)); % L integer

phase_ind = 2;

p_sample_initCont = NaN(1 + (nSamples-1)*maxiter,nd); % param samples
ss_sample_initCont = NaN(1 + (nSamples-1)*maxiter,ns); % sum-of-square samples
s2_sample_initCont = NaN(1 + (nSamples-1)*maxiter,ns); % sigma2 samples from the sampling phase

nESJD = NaN(maxiter,1);

% Initialise position vector with last param values from the initial design
p_sample_initCont(1,:) = p_sample_init(end,:);
ss_sample_initCont(1,:) = ss_sample_init(end,:);
s2_sample_initCont(1,:) = s2_sample_init(end,:);

% Also refine LogPost and Grads for the new s2
em_ind = 0; grad1_SimInd = 0; grad23_EmInd = [NaN NaN];
[LogPosterior_sim, ~, ~, ss_sample_initCont(1,:)] = ...
    HMCDerivPosterior_all_DCM(p_sample_initCont(1,:), ...
    s2_sample_initCont(1,:), trueData, ...
    sigma, nd,  em_ind, phase_ind, ...
    grad1_SimInd, grad23_EmInd, gp_regr_refitted, x_regr_refitted, ...
    y_regr_refitted, mean_y, std_y, do_nuts, invLref, IS, Ep, M, U, V, row, column);

noODE_counter_preproc = noODE_counter_preproc + 1;

em_ind = 1; grad1_SimInd = NaN; grad23_EmInd = [0 0];
[LogPosterior_em,GradLogPost_em] = ...
    HMCDerivPosterior_all_DCM(p_sample_initCont(1,:), ...
    s2_sample_initCont(1,:), trueData, sigma, nd, em_ind, phase_ind, grad1_SimInd, ...
    grad23_EmInd, gp_regr_refitted, x_regr_refitted, y_regr_refitted, ...
    mean_y, std_y, do_nuts, invLref, IS, Ep, M, U, V, row, column);

nstarts = 4; % for the optimisation of the acquisition function

nESJD_augm = NaN(maxiter+np-1,1);
x_E_augm = NaN(maxiter+np-1,2);

nESJD_augm(1:np-1) = nESJD_init(1:np-1); % this is unscaled
x_E = [epsilon_init(1:np-1), L_init(1:np-1)]; % (np,2) size
x_E_augm(1:np-1,:) = x_E; % already scaled

next = 1; % keeps track of the iteration no in the Markov chain for the ODE param
ind = np-1; % keeps track of the iteration no in the list of training points for nESJD
acc = 0;

initime = cputime();

tic()

rng('shuffle')

delete(gcp('nocreate'))
parpool('local', nstarts)

for j = 1:maxiter
    % Run HMC nSample times for [epsilon_adapt(j), L_adapt(j)]
    for i = 2:nSamples
        % Call HMC_fast to get p_sample_initCont
        next = next + 1;
        disp(strcat('sampled ',num2str((i-1)*j),' points'))
        [p_sample_initCont(next,:),LogPosterior_sim,LogPosterior_em,...
            GradLogPost_em,ss_sample_initCont(next,:), noODE_counter_iter] = ...
            HMC_DCM(p_sample_initCont(next-1,:), ...
            s2_sample_initCont(next-1,:), epsilon_adapt(j), L_adapt(j), ...
            gp_regr_refitted, x_regr_refitted, y_regr_refitted, ...
            nd, phase_ind, trueData, sigma,  ...
            ss_sample_initCont(next-1,:), mean_y, std_y, do_nuts, ...
            Moment, invLref, do_DA, IS, Ep, M, U, V, row, column);

        noODE_counter_preproc = noODE_counter_preproc + noODE_counter_iter;

        if all(p_sample_initCont(next,:) ~= p_sample_initCont(next-1,:)) % i.e. we've just accepted the new point
            %disp('accept')
            acc = acc + 1;
        else
            %disp('reject')
        end

        s2_sample_initCont(next,:) = 1./gamrnd(a+0.5*span, 1./(b+0.5*ss_sample_initCont(next,:)));

    end % nSamples

    %If we have gathered np data points in the sampling phase for the GP,
    %discard the points from the burnin phase (np points) not to affect the inference
    if (j == np)
        nESJD_augm(1:np) = []; x_E_augm(1:np,:) = [];
        ind = ind - j;
    end

    % Obtain the objective function
    xj = p_sample_initCont(next-nSamples+1:next,:);
    nESJD(j) = ESJDfct(xj)/sqrt(L_adapt(j));

    % Augment data set with ([eps(j), L(j)], nESJD(j))
    ind = ind + 1;
    nESJD_augm(ind) = nESJD(j);
    x_E_augm(ind,:) = [epsilon_adapt(j), L_adapt(j)];

    % Re-train GP
    mean_y_E = mean(nESJD_augm(1:ind));
    std_y_E = std(nESJD_augm(1:ind));
    x_E = x_E_augm(1:ind,:); 
    y_E = (nESJD_augm(1:ind) - mean_y_E)./std_y_E;

    gp_E = gp_optim(gp_E,x_E,y_E); % refine cov hyperparameters

    % Conduct Bayesian Optimisation
    betaj = 2*log((pi^2*(j+1)^(hd/2+2))/(3*delta));
    pj = 1; % to recover the original upper confidence bound function
    s = 1; % we don't need the rescale parameter as we are already scaling the data to zero mean, variance 1
    fh_ucb = @(x_new) UpperConfBound(x_new, gp_E, x_E, y_E, ...
        mean_y_E, std_y_E, s, pj, betaj);
    % Set the options for optimizer of the acquisition function
    optimf = @fmincon;

    optdefault=struct('LargeScale','off','Display', 'off', ...
        'Algorithm','active-set','TolFun',1e-3,'TolX',1e-3, ...
        'GradObj','on','SpecifyObjectiveGradient',true);

    options = optimset(optdefault);

    x0_E = NaN(nstarts, hd);
    x_E_optims = NaN(nstarts, hd); f_E_optims = NaN(nstarts, 1);



    parfor s1 = 1:nstarts
        % initial epsilon and L for the optimization
        x0_E(s1,:) = l_HMC+(u_HMC-l_HMC) * rand;

        [x_E_optims(s1,:), f_E_optims(s1)] = ...
            optimf(fh_ucb, x0_E(s1,:), [], [], [], [], ...
            l_HMC, u_HMC, [], options);
    end

    I_min = find(f_E_optims==min(f_E_optims));
    x_E_optim = x_E_optims(I_min(1),:);
    f_E_optim = f_E_optims(I_min(1));

    % Store new optimum
    epsilon_adapt(j+1) = x_E_optim(1);
    L_adapt(j+1) = round(x_E_optim(2)); % L integer

end %maxiter

% Now find the 'best' epsilon and L (i.e. that have generated the highest nESJD)
bestEps = x_E_augm(nESJD_augm==max(nESJD_augm),1);
bestL = round(x_E_augm(nESJD_augm==max(nESJD_augm),2));


ntimeit=1;
%sampling phase
for it = 1:ntimeit
    
    do_DA = 0; % do delayed acceptance
    
    nSamples = 8000; % no of sampling phase samples
    nburnin = 2000; % no of burnin phase samples
    L = bestL; % no of steps in leapfrog scheme
    epsilon = bestEps; % step size in leapfrog scheme
    phase_ind = 2; % phase index in Rasmussen's paper
    
    Moment = 0.01*eye(nd); % mass matrix for momentum
    
    nrun = 10; % run 10 chains in parallel
    acc = zeros(nrun,1); % acceptance rate for every chain
    
    p_sample = cell(nrun,1); % parameter samples from the sampling phase
    s2_sample = cell(nrun,1); % sigma2 samples from the sampling phase
    ss_sample = cell(nrun,1); % sigma2 samples from the sampling phase
    
    noODE_counter_proc = zeros(nrun,1); % count no of ODE evaluations in the processing (sampling) phase
    
    delete(gcp('nocreate'))
    parpool('local', nrun)
    
    % Store cpu times for every run and average the at the end
    initime = NaN(nrun,1);
    fintime = NaN(nrun,1);
    
    % Run 10 chains in parallel from different initialisations and different
    % random seed generators
    
    parfor j=1:nrun
        
        noODE_counter_run = 0;
        
        % Initialise
        p_sample{j}(1,:) = x_regr_refitted(end-j,:); % unbounded
        
        ss_sample{j}(1,:) = Run_simulator_DCM(p_sample{j}(1,:),  ...
             trueData, IS, Ep, M, U, V, row, column);
        
        s2_sample{j}(1,:) = ss_sample{j}(1,:)/span;
        
        % Get initial log likelihood, log posterior and gradient of log posterior
        % w.r.t. every theta_i
        em_ind = 0; grad1_SimInd = 0; grad23_EmInd = [NaN NaN];
        [LogPosterior_sim, ~, ~, ss_sample{j}(1,:)] = ...
            HMCDerivPosterior_all_DCM(p_sample{j}(1,:), ...
            s2_sample{j}(1,:), trueData, ...
            sigma, nd, em_ind, ...
            phase_ind, grad1_SimInd, grad23_EmInd, gp_regr_refitted, ...
            x_regr_refitted, y_regr_refitted, mean_y, std_y, do_nuts, invLref,IS, Ep, M, U, V, row, column);
        
        em_ind = 1;
        grad1_SimInd = NaN;
        grad23_EmInd = [0 0];
        [LogPosterior_em,GradLogPost_em] = ...
            HMCDerivPosterior_all_DCM(p_sample{j}(1,:), ...
            s2_sample{j}(1,:), trueData, ...
            sigma, nd, em_ind, ...
            phase_ind, grad1_SimInd, grad23_EmInd, gp_regr_refitted, ...
            x_regr_refitted, y_regr_refitted, mean_y, std_y, do_nuts, invLref,IS, Ep, M, U, V, row, column);
        
        % Enter HMC loop to draw nSamples
        for i=2:nSamples+nburnin
            disp(strcat('explore ',num2str(i-1),'sample'))
            
            if i == nburnin + 1 % start measuring after the burnin
                initime(j) = cputime;
                tic;
            end
            
            [p_sample{j}(i,:),LogPosterior_sim,LogPosterior_em,...
                GradLogPost_em, ss_sample{j}(i,:), no_counter_iter] = ...
                HMC_DCM(p_sample{j}(i-1,:), s2_sample{j}(i-1,:), ...
                epsilon, ceil(rand*L), gp_regr_refitted, x_regr_refitted, y_regr_refitted, ...
                nd, phase_ind, trueData, sigma,  ...
                ss_sample{j}(i-1,:), mean_y, std_y, ...
                do_nuts, Moment, invLref, do_DA, IS, Ep, M, U, V, row, column);
            
            if i > nburnin % start counting in the sampling phase
                noODE_counter_run = noODE_counter_run + no_counter_iter;
            end
            
            %ss_sample{j}(i);
            
            if all(p_sample{j}(i,:) ~= p_sample{j}(i-1,:)) % we've just accepted the new point
                acc(j) = acc(j) + 1;
                %fprintf('accept for run %d \n',j)
            end
            
            
            if i<nburnin % sample sigma2 in sampling phase
                s2_sample{j}(i,:) = s2_sample{j}(i-1,:);
            else
                %disp('sampling phase')
                s2_sample{j}(i,:) = 1./gamrnd(a+0.5*T, 1./(b+0.5*ss_sample{j}(i,:)));
            end
            
            
        end
        
        noODE_counter_proc(j) = noODE_counter_proc(j) + noODE_counter_run;
        
        toc;
        
        fintime(j) = cputime;
    end
    
    CPUtime_BO_DAGPHMC_sampling = fintime-initime;
    
    ElapsedTime_BO_DAGPHMC_sampling = NaN(nd,1);
    
       
end % ntimeit

for i=1:nrun
    p_result(i,:)=p_sample{i,1}(end,:);
end

A_esti=cell(nrun,1);
for j=1:nrun 
    for i=1:size_row
        A_esti{j,1}(row(i),column(i))=p_result(j,i);
    end
end

for i=1:nrun
    rmse(i)=sqrt(sum(sum((A_esti{i,1}-At).^2))/(n*n));
end