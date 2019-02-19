function [models,figHandle] = fLoopSubSpace(freq,G,covG,na,max_r,optimize,forcestability,showfigs,fs)
%FLOOPSUBSPACE Loop frequency-domain subspace method over multiple model orders and sizes of the extended observability matrix.
%
%	Usage:
%		[models,figHandle] = fLoopSubSpace(freq,G,covG,na,max_r,optimize,forcestability,showfigs,fs)
%       models = fLoopSubSpace(freq,G,covG,na,max_r,optimize)
%
%	Description:
%		fLoopSubSpace estimates linear state-space models from samples of
%		the frequency response function (or frequency response matrix). The
%		frequency-domain subspace method in McKelvey et al. (1996) is
%		applied with the frequency weighting in Pintelon (2002). The
%		subspace algorithm is looped over different model orders and
%		different block rows of the extended observability matrix. If
%		requested, a Levenberg-Marquardt algorithm is carried out to
%		optimize the model parameters obtained with the subspace algorithm.
%		Stability of the estimated models can be forced (to some extent).
%       For each model order, the model that minimizes the weighted
%       mean-square error
%
%             F
%            ----
%          1 \
%      e = - /   real(vec(G(f) - Gm(f))'*covGinv(f)*vec(G(f) - Gm(f)))
%          F ----
%            f = 1
%
%       is retained as the best model for that order. Here, Gm(f) is the
%       FRF (or FRM) of the (optimized) subspace model.
%
%	Output parameters:
%		models : cell array where models{n} contains a 1 x 4 cell with the
%		         A, B, C, and D matrices of the best model of order n (see
%		         the Description for more information on 'best')
%       figHandle : figure handle of the last plotted figure
%
%	Input parameters:
%		freq : vector of frequencies at which the FRF (or FRM) is given (in
%              Hz and in the interval [0, fs/2])
%		G : m x p x F array of FRF (or FRM) data
%		covG : m*p x m*p x F covariance tensor on G
%              (0 if no weighting required)
%		na : vector of model orders to scan
%		max_r : maximum number of block rows in the extended observability
%		        matrix (no models are estimated for orders n >= max_r)
%		optimize : number of Levenberg-Marquardt optimizations
%                  (0 if no optimization required, maximum 1000)
%		forcestability : boolean indicating whether or not a stable model
%		                 is forced (optional, default = true)
%		showfigs : boolean indicating whether or not to show figures
%		           (optional, default = true)
%		fs : sampling frequency (in Hz)
%            (optional, default = 1 Hz)
%
%	Example:
%       n = 2; % Model order
%       N = 1000; % Number of samples
%       fs = 1; % Sampling frequency
%       sys = drss(n); % Random state-space model
%       freq = (0:floor(N/2)-1)*fs/N; % Frequency vector
%       G = fss2frf(sys.A,sys.B,sys.C,sys.D,freq/fs); % FRF
%       covG = 0; % No weighting
%       max_r = n+3; % Maximal number of block rows in the extended observability matrix
%       optimize = 0; % No Levenberg-Marquardt optimization loops
%       models = fLoopSubSpace(freq,G,covG,n,max_r,optimize);
%       model = models{n}; % Select nth-order model
%       sys_est = ss(model{:},1/fs); % Estimated state-space model
%       figure
%       bode(sys,'b-',sys_est,'r--') % Compare Bode plots of both models
%       legend('True','Estimated')
%
%	References:
%		McKelvey T., Akcay, H., and Ljung, L. (1996). Subspace-Based
%       Multivariable System Identification From Frequency Response Data.
%       IEEE Transactions on Automatic Control, 41(7):960-979.
%
%       Pintelon, R. (2002). Frequency-domain subspace system
%       identification using non-parametric noise models. Automatica,
%       38:1295-1311.
%
%	Versions:
%		1.0 : August 13, 2015
%       1.1 : August 19, 2015
%           Cost function subspace models calculated only once.
%           Cost of stabilized state space model used instead of cost of
%           unstable model that was optimized.
%           Variable 'temp' contains best stable/stabilized subspace model.
%           Stability checking before assigning non-optimized model removed
%           (if stability is forced, each subspace model is either stable 
%           or stabilized).
%           Renamed variable 'unstablesSS' to 'stabilizedSS'.
%           Start from same color in Summary plot for newest Matlab
%           versions.
%       1.2 : April 20, 2016
%           Help updated
%
%	Copyright (c) Vrije Universiteit Brussel – dept. ELEC
%   All rights reserved.
%   Software can be used freely for non-commercial applications only.
%   Disclaimer: This software is provided “as is” without any warranty.
%
%   See also fFreqDomSubSpace, fss2frf, fStabilize, fVec, fLevMarqFreqSSz, fIsUnstable, fPlotFrfMIMO

%--------------------------------------------------------------------------
% Version 1.1 = Version 1.2
%   Cost function subspace models calculated only once.
%   Cost of stabilized state space model used instead of cost of unstable
%   model that was optimized.
%   Variable 'temp' contains best subspace model ('models{n} gets assigned
%   'temp' only if stability is forced, if optimization is done, and if all
%   optimized models are unstable, so that in those cases 'temp' indeed
%   contains the best stable/stabilized subspace model).
%   Stability checking before assigning non-optimized model removed (if
%   stability is forced, each subspace model is either stable or
%   stabilized).
%   Renamed variable 'unstablesSS' to 'stabilizedSS'.
%   Start from same color in Summary plot for newest Matlab versions.
%--------------------------------------------------------------------------
% {

% Set default values if some input arguments are not specified
if (nargin < 9) || isempty(fs)
    fs = 1; % By default, the sampling frequency is 1 Hz
end
if (nargin < 8) || isempty(showfigs)
    showfigs = true; % By default, figures are shown
end
if (nargin < 7) || isempty(forcestability)
    forcestability = true; % By default, a stable model is forced
end

min_na = min(na); % Minimal model order that is scanned
max_na = max(na); % Maximal model order that is scanned
% Preallocate some variables
KSSs = NaN(max_na,max_r); % Cost functions subspace algorithm
KLMs = NaN(max_na,max_r); % Cost functions Levenberg-Marquardt (LM) optimization
stabilizedSS = zeros(size(KSSs)); % Matrix indicating stabilized models from subspace algorithm
unstablesLM = zeros(size(KLMs)); % Matrix indicating unstable models from LM optimization
models = cell(max_na,1); % Cell containing the model parameters of the best models for each scanned order

% Determine number of outputs/inputs and number of frequencies
[p,m,F] = size(G);

% Compute inverse of covariance matrix for weighting
if covG == 0 % No weighting info is available
    covGinv = repmat(eye(m*p),[1 1 F]);
else
    covGinv = zeros(m*p,m*p,F);
    for f = 1:F
        covGinv(:,:,f) = pinv(covG(:,:,f));
    end
end

h = waitbar(0,'Please wait...');

for n = na % Scan all model orders
    min_r = n + 1;
    for r = min_r:max_r % Scan all r values
        disp(['n = ' num2str(n) ' r = ' num2str(r)])
        [A,B,C,D,unstable] = fFreqDomSubSpace(G,covG,freq/fs,n,r); % Frequency-domain subspace algorithm
        if unstable && ~forcestability
            unstable = false; % If stability is not forced, 'unstable' is always false
        end;
        if forcestability && unstable
            [A,B,C,D] = fStabilize(A,B,C,D,'z');
            stabilizedSS(n,r) = 1; % Mark this model as stabilized
        end
        GSS = fss2frf(A,B,C,D,freq/fs); % FRF of the subspace model
        KSS = 0;
        errSS = G - GSS;
        for i = 1:F % Compute cost function subspace model
            KSS = real(fVec(errSS(:,:,i))'*covGinv(:,:,i)*fVec(errSS(:,:,i))) + KSS;
        end
        KSS = KSS/F; % Normalize with the number of excited frequencies
        KSSs(n,r) = KSS; % Store the cost function of the subspace algorithm
      
        if optimize > 0 % If Levenberg-Marquardt optimizations requested
            [ALM,BLM,CLM,DLM] = fLevMarqFreqSSz(freq/fs,G,covG,A,B,C,D,optimize); % Optimize model parameters
            GLM = fss2frf(ALM,BLM,CLM,DLM,freq/fs);
            KLM = 0;
            errLM = G - GLM;
            for i = 1:F % Compute cost function optimized model
                KLM = real(fVec(errLM(:,:,i))'*covGinv(:,:,i)*fVec(errLM(:,:,i))) + KLM;
            end
            KLM = KLM/F; % Normalize with the number of excited frequencies
            KLMs(n,r) = KLM; % Save the cost function of the LM optimization
            
            if min(KSSs(n,:)) == KSS % If the current subspace model achieves the smallest cost so far ...
                temp = {A,B,C,D}; % ... temporarily save the parameters of that model
            end
            if forcestability && fIsUnstable(ALM,'z')
                unstablesLM(n,r) = 1; % Mark this model as unstable
            else
                if min(KLMs(n,~logical(unstablesLM(n,:)))) == KLM % If the current optimized model achieves the smallest cost of all stable optimized models so far ...
                    models{n} = {ALM,BLM,CLM,DLM}; % ... save the parameters of the current model
                end
            end
            if all(unstablesLM(n,n+1:max_r) == 1) % If all optimized models are unstable and stability is forced, ...
                models{n} = temp; % ... take the parameters of the best non-optimized subspace model so far (stable or stabilized)
            end
        else % No optimization
            if min(KSSs(n,:)) == KSS % If it is the smallest and stable/stabilized (or unstable, but stability not forced)
                models{n} = {A,B,C,D};
            end
        end
        waitbar(((n - min_na) + (r - min_r)/(max_r - min_r))/(max_na - min_na + 1),h);
    end
end
close(h);

% Remove entries corresponding to model orders that were not scanned
KSSs(1:min_na - 1,:) = [];
KLMs(1:min_na - 1,:) = [];
stabilizedSS(1:min_na - 1,:) = [];
unstablesLM(1:min_na - 1,:) = [];
% Retain cost functions of unstable models, and put those of stable models to 0 (= -Inf on a logarithmic scale)
stabilizedSS = KSSs.*stabilizedSS;
unstablesLM = KLMs.*unstablesLM;

if showfigs
    figure
    semilogy(KSSs','.') % Cost function of all subspace models
    hold on;
    semilogy(stabilizedSS','o','Color',[0.5 0.5 0.5]) % Encircle stabilized models in gray
    if optimize > 0
        try
            set(gca,'ColorOrderIndex',1) % Restart from the same color as in the KSSs plot
        catch % Older Matlab versions restart by default from the same color, and 'ColorOrderIndex' is not available
        end
        semilogy(KLMs','*') % Cost function of all LM-optimized models
        try
            set(gca,'ColorOrderIndex',1) % Restart from the same color as in the KSSs plot
        catch % Older Matlab versions restart by default from the same color, and 'ColorOrderIndex' is not available
        end
        semilogy(unstablesLM','o') % Encircle unstable models
    end
    ylabel('V_{WLS}')
    xlabel('r')
    legend(cellstr([repmat('n = ',max_na - min_na + 1,1) num2str((min_na:max_na)')]));
    title({'Cost functions of subspace models (dots, stablilized models encircled in gray)', ...
           'and of LM-optimized models (stars, unstable models encircled in color)'})
    xlim([min_na (max_r + 1)])
    set(gcf,'Name','Summary')
    set(gcf,'NumberTitle','off')
    
    for n = na % Plot FRFs of the best model for each model order
        figHandle = figure;
        model = models{n};
        try % Maybe no stable models were estimated
            A = model{1};
            B = model{2};
            C = model{3};
            D = model{4};
            GModel = fss2frf(A,B,C,D,freq/fs);
            fPlotFrfMIMO(GModel,freq);
            fPlotFrfMIMO(G,freq,'b.');
            fPlotFrfMIMO(G-GModel,freq,'r.');
            temp = zeros(m*p,F);
            for i = 1:m*p
                temp(i,:) = covG(i,i,:);
            end
            temp2 = zeros(size(G));
            for f = 1:F
                temp2(:,:,f) = reshape(temp(:,f),p,m);
            end
            stdG = sqrt(temp2);
            fPlotFrfMIMO(stdG,freq,'k--');
            set(gcf,'Name',['n = ' num2str(n)])
            set(gcf,'NumberTitle','off')
            
            legend('BLA (parametric)','BLA (nonpar)','residual','standard deviation')
        catch
            gca;
            title('No stable models');
        end
    end
else
    figHandle = [];
end;

%}


%--------------------------------------------------------------------------
% Version 1.0
%--------------------------------------------------------------------------
%{

% Set default values if some input arguments are not specified
if nargin < 9
    fs = 1;
end
if nargin < 8
    showfigs = true; % By default, figures are shown
end
if nargin < 7
    forcestability = true; % By default, a stable model is forced
end

min_na = min(na); % Minimal model order that is scanned
max_na = max(na); % Maximal model order that is scanned
% Preallocate some variables
KSSs = NaN(max_na,max_r); % Cost functions subspace algorithm
KLMs = NaN(max_na,max_r); % Cost functions Levenberg-Marquardt (LM) optimization
unstablesSS = zeros(size(KSSs)); % Matrix indicating stabilized models from subspace algorithm
unstablesLM = zeros(size(KLMs)); % Matrix indicating unstable models from LM optimization
models = cell(max_na,1); % Cell containing the model parameters of the best models for each scanned order

% Determine number of inputs/outputs and number of frequencies
[p,m,F] = size(G);

% Compute inverse of covariance matrix for weighting
if covG == 0 % No weighting info is available
    covGinv = repmat(eye(m*p),[1 1 F]);
else
    covGinv = zeros(m*p,m*p,F);
    for f = 1:F
        covGinv(:,:,f) = pinv(covG(:,:,f));
    end
end

h = waitbar(0,'Please wait...');

for n = na % Scan all model orders
    min_r = n + 1;
    for r = min_r:max_r % Scan all r values
        disp(['n = ' num2str(n) ' r = ' num2str(r)])
        [A,B,C,D,unstable] = fFreqDomSubSpace(G,covG,freq/fs,n,r); % Frequency-domain subspace algorithm
        if unstable && ~forcestability
            unstable = false; % If stability is not forced, 'unstable' is always false
        end;
        GSS = fss2frf(A,B,C,D,freq/fs); % FRF of the subspace model
        KSS = 0;
        errSS = G - GSS;
        if forcestability && unstable
            [A,B,C,D] = fStabilize(A,B,C,D,'z');
            unstablesSS(n,r) = 1; % Mark this model as stabilized
        end
      
        if optimize > 0 % If Levenberg-Marquardt optimizations requested
            [ALM,BLM,CLM,DLM] = fLevMarqFreqSSz(freq/fs,G,covG,A,B,C,D,optimize); % Optimize model parameters
            GLM = fss2frf(ALM,BLM,CLM,DLM,freq/fs);
            KLM = 0;
            errLM = G - GLM;
            for i = 1:F % Cost functions are sums over the frequencies of weighted squared errors
                KSS = real(fVec(errSS(:,:,i))'*covGinv(:,:,i)*fVec(errSS(:,:,i))) + KSS;
                KLM = real(fVec(errLM(:,:,i))'*covGinv(:,:,i)*fVec(errLM(:,:,i))) + KLM;
            end
            KSS = KSS/F; % Normalize with the number of excited frequencies
            KLM = KLM/F; % Normalize with the number of excited frequencies
            KLMs(n,r) = KLM; % Save the cost function of the LM optimization
            KSSs(n,r) = KSS; % Save the cost function of the subspace algorithm
            
            if min(KSSs(n,:)) == KSS % If the current model achieves the smallest cost so far ...
                temp = {ALM,BLM,CLM,DLM}; % ... temporarily save the parameters of the current model
            end
            if forcestability && fIsUnstable(ALM,'z')
                unstablesLM(n,r) = 1; % Mark this model as unstable
            else
                if min(KLMs(n,~logical(unstablesLM(n,:)))) == KLM % If the current optimized model achieves the smallest cost of all stable optimized models so far ...
                    models{n} = {ALM,BLM,CLM,DLM}; % ... save the parameters of the current model
                end
            end
            if all(unstablesLM(n,n+1:max_r) == 1) % If all optimized models are unstable, ...
                models{n} = temp; % ... take the parameters of the best non-optimized subspace model so far (even if that model would be unstable)
            end
        else % No optimization
            for i = 1:F % Compute cost function
                KSS = real(fVec(errSS(:,:,i))'*covGinv(:,:,i)*fVec(errSS(:,:,i))) + KSS;
            end
            KSS = KSS/F;
            KSSs(n,r) = KSS;
            if min(KSSs(n,:)) == KSS && ~unstable % If it is the smallest and stable (or unstable, but stability not forced)
                models{n} = {A,B,C,D};
            end
        end
        waitbar(((n - min_na) + (r - min_r)/(max_r - min_r))/(max_na - min_na + 1),h);
    end
end
close(h);

% Remove entries corresponding to model orders that were not scanned
KSSs(1:min_na - 1,:) = [];
KLMs(1:min_na - 1,:) = [];
unstablesSS(1:min_na - 1,:) = [];
unstablesLM(1:min_na - 1,:) = [];
% Retain cost functions of unstable models, and put those of stable models to 0 (= -Inf on a logarithmic scale)
unstablesSS = KSSs.*unstablesSS;
unstablesLM = KLMs.*unstablesLM;

if showfigs
    figure
    semilogy(KSSs','.') % Cost function of all subspace models
    hold on;
    semilogy(unstablesSS','o','Color',[0.5 0.5 0.5]) % Encircle stabilized models in gray
    if optimize > 0
        set(gca,'ColorOrderIndex',1) % Restart from the same color as in the KSSs plot
        semilogy(KLMs','*') % Cost function of all LM-optimized models
        set(gca,'ColorOrderIndex',1) % Restart from the same color as in the KSSs plot
        semilogy(unstablesLM','o') % Encircle unstable models
    end
    ylabel('V_{WLS}')
    xlabel('r')
    legend(cellstr([repmat('n = ',max_na - min_na + 1,1) num2str((min_na:max_na)')]));
    title({'Cost functions of subspace models (dots, stablilized models encircled in gray)', ...
           'and of LM-optimized models (stars, unstable models encircled in color)'})
    xlim([min_na (max_r + 1)])
    set(gcf,'Name','Summary')
    set(gcf,'NumberTitle','off')
    
    for n = na % Plot FRFs of the best model for each model order
        figHandle = figure;
        model = models{n};
        try % Maybe no stable models were estimated
            A = model{1};
            B = model{2};
            C = model{3};
            D = model{4};
            GModel = fss2frf(A,B,C,D,freq/fs);
            fPlotFrfMIMO(GModel,freq);
            fPlotFrfMIMO(G,freq,'b.');
            fPlotFrfMIMO(G-GModel,freq,'r.');
            temp = zeros(m*p,F);
            for i = 1:m*p
                temp(i,:) = covG(i,i,:);
            end
            temp2 = zeros(size(G));
            for f = 1:F
                temp2(:,:,f) = reshape(temp(:,f),p,m);
            end
            stdG = sqrt(temp2);
            fPlotFrfMIMO(stdG,freq,'k--');
            set(gcf,'Name',['n = ' num2str(n)])
            set(gcf,'NumberTitle','off')
            
            legend('BLA (parametric)','BLA (nonpar)','residual','standard deviation')
        catch
            gca;
            title('No stable models');
        end
    end
else
    figHandle = [];
end;

%}