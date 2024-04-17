function [y_csd,IS,Ep,M,U,V] = spm_dcm_fmri_csd_MCMC(P)

if isstruct(P)
    DCM = P;
    try, DCM.name; catch, DCM.name = sprintf('DCM_%s',date); end
    P   = DCM.name;
else
    load(P)
end

% check options and specification
%==========================================================================
try, DCM.options.two_state;  catch, DCM.options.two_state  = 0;     end
try, DCM.options.stochastic; catch, DCM.options.stochastic = 0;     end
try, DCM.options.centre;     catch, DCM.options.centre     = 0;     end
try, DCM.options.analysis;   catch, DCM.options.analysis   = 'CSD'; end
try, DCM.options.order;      catch, DCM.options.order      = 8;     end
try, DCM.options.nograph;    catch, DCM.options.nograph    = spm('CmdLine');  end


% parameter initialisation
%--------------------------------------------------------------------------
try, DCM.M.P = DCM.options.P; end

% check max iterations
%--------------------------------------------------------------------------
try
    DCM.options.maxit;
catch    
    if isfield(DCM.options,'nN')
        DCM.options.maxit = DCM.options.nN;
        warning('options.nN is deprecated. Please use options.maxit');
    else
        DCM.options.maxit = 128;
    end
end
 
try DCM.M.Nmax; catch, DCM.M.Nmax = DCM.options.maxit; end

% check max nodes
%--------------------------------------------------------------------------
try
    DCM.options.maxnodes;
catch
    if isfield(DCM.options,'nmax')
        DCM.options.maxnodes = DCM.options.nmax;
        warning('options.nmax is deprecated. Please use options.maxnodes');
    else
        DCM.options.maxnodes = 8;
    end
end

% sizes
%--------------------------------------------------------------------------
try, DCM.U.u; catch, DCM.U.u = [];            end
try, DCM.n;   catch, DCM.n = size(DCM.a,1);   end
try, DCM.v;   catch, DCM.v = size(DCM.Y.y,1); end


% analysis and options
%--------------------------------------------------------------------------
DCM.options.induced    = 1;
DCM.options.nonlinear  = 0;
DCM.options.stochastic = 0;


% organise response variables: detrend outputs (and inputs)
%==========================================================================
DCM.Y.y = spm_detrend(DCM.Y.y);
if DCM.options.centre
    DCM.U.u = spm_detrend(DCM.U.u);
end

% scale timeseries to a precision of four
%--------------------------------------------------------------------------
vY          = spm_vec(DCM.Y.y);
scale       = 1/std(vY)/4;
DCM.Y.y     = spm_unvec(vY*scale,DCM.Y.y);
DCM.Y.scale = scale;

% disable high order parameters and check for models with no inputs
%--------------------------------------------------------------------------
n       = DCM.n;
if iscell(DCM.Y.y)
    
    % augment with between session constraints
    %----------------------------------------------------------------------
    if size(DCM.b,3) ~= numel(DCM.Y.y)
        for i = 1:numel(DCM.Y.y)
            DCM.b(:,:,i)  = DCM.b(:,:,1);
        end
    end
else
    DCM.b      = zeros(n,n,0);
end
if isempty(DCM.c) || isempty(DCM.U.u)
    DCM.c      = zeros(n,0);
    DCM.U.u    = zeros(DCM.v,1);
    DCM.U.name = {'null'};
end
DCM.d   = zeros(n,n,0);


% priors (and initial states)
%==========================================================================
[pE,pC,x] = spm_dcm_fmri_priors(DCM.a,DCM.b,DCM.c,DCM.d,DCM.options);

% eigenvector constraints on pC for large models
%--------------------------------------------------------------------------
if n > DCM.options.maxnodes
    
    % remove confounds and find principal (maxnodes) modes
    %----------------------------------------------------------------------
    try
        y   = DCM.Y.y - DCM.Y.X0*(pinv(DCM.Y.X0)*DCM.Y.y);
    catch
        y   = spm_detrend(DCM.Y.y);
    end
    V       = spm_svd(y');
    V       = V(:,1:DCM.options.maxnodes);
    
    % remove minor modes from priors on A
    %----------------------------------------------------------------------
    j       = 1:(n*n);
    V       = kron(V*V',V*V');
    pC(j,j) = V*pC(j,j)*V';
    
end

% place eigenmodes in model if DCM.a is a vector (of eigenvalues)
%--------------------------------------------------------------------------
if isvector(DCM.a)
    DCM.M.modes = spm_svd(cov(DCM.Y.y));
end

% check for pre-specified priors
%--------------------------------------------------------------------------
hE       = 8;
hC       = 1/128;
try, pE  = DCM.M.pE; pC  = DCM.M.pC; end
try, hE  = DCM.M.hE; hC  = DCM.M.hC; end

% create DCM
%--------------------------------------------------------------------------
DCM.M.IS = 'spm_csd_fmri_mtf';
DCM.M.g  = @spm_gx_fmri;
DCM.M.f  = @spm_fx_fmri;
DCM.M.x  = x;
DCM.M.pE = pE;
DCM.M.pC = pC;
DCM.M.hE = hE;
DCM.M.hC = hC;
DCM.M.n  = length(spm_vec(x));
DCM.M.m  = size(DCM.U.u,2);
DCM.M.l  = n;
DCM.M.p  = DCM.options.order;

% specify M.u - endogenous input (fluctuations) and intial states
%--------------------------------------------------------------------------
DCM.M.u  = sparse(n,1);

% get data-features (MAR(p) model)
%==========================================================================
DCM.Y.p  = DCM.M.p;
DCM      = spm_dcm_fmri_csd_data(DCM);
DCM.M.Hz = DCM.Y.Hz;
DCM.M.dt = 1/2;
DCM.M.N  = 32;
DCM.M.ns = 1/DCM.Y.dt;


% scale input (to a variance of 1/8)
%--------------------------------------------------------------------------
if any(diff(DCM.U.u))
    ccf         = spm_csd2ccf(DCM.U.csd,DCM.Y.Hz);
    DCM.U.scale = max(spm_vec(ccf))*8;
    DCM.U.csd   = spm_unvec(spm_vec(DCM.U.csd)/DCM.U.scale,(DCM.U.csd));
end


% complete model specification and invert
%==========================================================================

% precision of spectral observation noise
%--------------------------------------------------------------------------
%DCM.Y.Q  = spm_dcm_csd_Q(DCM.Y.csd);
DCM.Y.X0 = sparse(size(DCM.Y.csd,1)*size(DCM.Y.csd,2)*size(DCM.Y.csd,3),0);
DCM.Y.p  = DCM.M.p;

% Variational Laplace: model inversion (using spectral responses)
%==========================================================================
Y.y          = DCM.Y.csd;
y_csd = Y.y;
[IS,Ep,M,U,V] = spm_nlsi_GN_MCMC(DCM.M,DCM.U,Y);

function [IS,Ep,M,U,V] = spm_nlsi_GN_MCMC(M,U,Y)

try
    y  = Y.y;
catch
    y  = Y;
end

try
    
    % try FS(y,M)
    %----------------------------------------------------------------------
    try
        y  = feval(M.FS,y,M);
        IS = inline([M.FS '(' M.IS '(P,M,U),M)'],'P','M','U');
        
        % try FS(y)
        %------------------------------------------------------------------
    catch
        y  = feval(M.FS,y);
        IS = inline([M.FS '(' M.IS '(P,M,U))'],'P','M','U');
    end
    
catch
    
    % otherwise FS(y) = y
    %----------------------------------------------------------------------
    try
        IS = inline([M.IS '(P,M,U)'],'P','M','U');
    catch
        IS = M.IS;
    end
end

% converted to function handle
%--------------------------------------------------------------------------
IS  = spm_funcheck(IS);

% paramter update eqation
%--------------------------------------------------------------------------
if isfield(M,'f'), M.f = spm_funcheck(M.f);  end
if isfield(M,'g'), M.g = spm_funcheck(M.g);  end
if isfield(M,'h'), M.h = spm_funcheck(M.h);  end


% size of data (samples x response component x response component ...)
%--------------------------------------------------------------------------
if iscell(y)
    ns = size(y{1},1);
else
    ns = size(y,1);
end
ny   = length(spm_vec(y));          % total number of response variables
nr   = ny/ns;                       % number response components
M.ns = ns;                          % number of samples M.ns

% initial states
%--------------------------------------------------------------------------
try
    M.x;
catch
    if ~isfield(M,'n'), M.n = 0;    end
    M.x = sparse(M.n,1);
end

% input
%--------------------------------------------------------------------------
try
    U;
catch
    U = [];
end

% initial parameters
%--------------------------------------------------------------------------
try
    spm_vec(M.P) - spm_vec(M.pE);
    fprintf('\nParameter initialisation successful\n')
catch
    M.P = M.pE;
end

% precision components Q
%--------------------------------------------------------------------------
try
    Q = Y.Q;
    if isnumeric(Q), Q = {Q}; end
catch
    Q = spm_Ce(ns*ones(1,nr));
end
nh    = length(Q);                  % number of precision components


% prior moments (assume uninformative priors if not specifed)
%--------------------------------------------------------------------------
pE       = M.pE;
try
    pC   = M.pC;
catch
    np   = spm_length(M.pE);
    pC   = speye(np,np)*exp(16);
end

% confounds (if specified)
%--------------------------------------------------------------------------
try
    nb   = size(Y.X0,1);            % number of bins
    nx   = ny/nb;                   % number of blocks
    dfdu = kron(speye(nx,nx),Y.X0);
catch
    dfdu = sparse(ny,0);
end
if isempty(dfdu), dfdu = sparse(ny,0); end


% unpack covariance
%--------------------------------------------------------------------------
if isstruct(pC);
    pC = spm_diag(spm_vec(pC));
end

% dimension reduction of parameter space
%--------------------------------------------------------------------------
V     = spm_svd(pC,0);               % number of parameters (confounds)
np    = size(V,2);                    % number of parameters (effective)
ip    = (1:np)';


% initialize conditional density
%--------------------------------------------------------------------------
Eu    = spm_pinv(dfdu)*spm_vec(y);
p     = [V'*(spm_vec(M.P) - spm_vec(M.pE)); Eu];
Ep    = spm_unvec(spm_vec(pE) + V*p(ip),pE);

