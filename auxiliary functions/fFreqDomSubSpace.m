function varargout = fFreqDomSubSpace(H,covarH,freq,n,r)
%FFREQDOMSUBSPACE Estimate state-space model from Frequency Response Function (or Matrix).
%
%	Usage:
%		[A,B,C,D,unstable] = fFreqDomSubSpace(H,covarH,freq,n,r)
%       [A,B,C,D] = fFreqDomSubSpace(H,covarH,freq,n,r)
%
%	Description:
%		fFreqDomSubSpace estimates linear state-space models from samples
%		of the frequency response function (or frequency response matrix). 
%		The frequency-domain subspace method in McKelvey et al. (1996) is
%		applied with the frequency weighting in Pintelon (2002), i.e.
%		weighting with the sampled covariance matrix.
%
%	Output parameters:
%		A : n x n state matrix
%       B : n x m input matrix
%       C : p x n output matrix
%       D : p x m feed-through matrix
%       unstable : boolean indicating whether or not the identified
%                  state-space model is unstable
%
%	Input parameters:
%		H : p x m x F Frequency Response Matrix (FRM)
%       covarH : p*m x p*m x F covariance tensor on H
%                (0 if no weighting required)
%       freq : vector of normalized frequencies at which the FRM is given
%              (0 < freq < 0.5)
%       n : model order
%       r : number of block rows in the extended observability matrix
%           (r > n)
%
%	Example:
%       n = 2; % Model order
%       N = 1000; % Number of samples
%       fs = 1; % Sampling frequency
%       sys = drss(n); % Random state-space model
%       freq = (0:floor(N/2)-1)*fs/N; % Frequency vector
%       H = fss2frf(sys.A,sys.B,sys.C,sys.D,freq/fs); % FRF
%       covarH = 0; % No weighting
%       r = n+1; % Maximal number of block rows in the extended observability matrix
%       [A,B,C,D] = fFreqDomSubSpace(H,covarH,freq/fs,n,r);
%       sys_est = ss(A,B,C,D,1/fs); % Estimated state-space model
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
%       Paduart J. (2008). Identification of nonlinear systems using
%       polynomial nonlinear state space models. PhD thesis, Vrije
%       Universiteit Brussel.
%
%	Versions:
%		1.0 : August 20, 2015
%       1.1 : April 19, 2016
%           Help updated
%
%	Copyright (c) Vrije Universiteit Brussel – dept. ELEC
%   All rights reserved.
%   Software can be used freely for non-commercial applications only.
%   Disclaimer: This software is provided “as is” without any warranty.
%
%   See also fSqrtInverse, fJacobFreqSS, fWeightJacobSubSpace, fNormalizeColumns, fIsUnstable  

%--------------------------------------------------------------------------
% Version 1.0 = Version 1.1
%--------------------------------------------------------------------------
% {

%   Algorithm:
%       1. Construct Extended observability matrix Or
%           a. Construct Wr with z
%           b. Construct Hbold with H and Wr
%           c. Construct Ubold with Wr (U=eye(m))
%           d. Split real and imaginary parts of Ubold and Hbold
%           e. Z=[Ubold;Hbold];
%           f. Calculate CY
%           g. QR decomposition of Z.'
%           h. CY^(-1/2)*RT22=USV'
%           i. Or=U(:,1:n)
%       2. Estimate A and C with Or
%       3. Estimate B and D given A,C and H

% Determine number of outputs/inputs and number of frequencies
[p,m,F] = size(H);

% 1.a. Construct Wr with z
z = exp(1j*2*pi*freq(:));
Wr = (repmat(z,1,r).^repmat(0:r-1,length(z),1)).';

% 1.b. and 1.c. Construct Hbold and Ubold
Hbold = zeros(r*p,F*m);
Ubold = zeros(r*m,F*m);
for f=1:F
    % 1.b. Construct Hbold with H and Wr
    Hbold(:,(f-1)*m+1:f*m) = kron(Wr(:,f),H(:,:,f));
    % 1.c. Construct Ubold with Wr (U=eye(m))
    Ubold(:,(f-1)*m+1:f*m) = kron(Wr(:,f),eye(m));
end

% 1.d. Split real and imaginary parts of Ubold and Hbold
Hbold = [real(Hbold) imag(Hbold)];
Ubold = [real(Ubold) imag(Ubold)];

% 1.e. Z=[Ubold;Hbold];
Z  = [Ubold; Hbold];
clear Hbold Ubold;

% 1.f. Calculate CY
if covarH == 0
    CY = eye(length(kron(Wr(:,1)*Wr(:,1)',eye(p))));
    covarH = repmat(eye(p*m),[1 1 F]);
else
    CY = 0;
    for f = 1:F
        % Take sum over the diagonal blocks of cov(vec(H)) (see thesis
        % Johan Paduart, (5-93))
        temp = zeros(p,p);
        for i = 0:m-1
            temp = temp + covarH(i*p+1:(i+1)*p,i*p+1:(i+1)*p,f);
        end
        CY = CY + real(kron(Wr(:,f)*Wr(:,f)',temp));
    end
end

% 1.g. QR decomposition of Z.'
[~,R] = qr(Z.',0); % Economy size QR factorization
RT = R.';
RT22 = RT(end-(r*p)+1:end,end-(r*p)+1:end);

% 1.h. CY^(-1/2)*RT22=USV'
    % Calculate CY^(-1/2)
[uc,sc] = svd(CY,0);
minsqrtCY = uc*diag(diag(sc).^(-1/2))*uc';
sqrtCY = uc*sqrt(sc)*uc';
    % Take SVD of CY^(-1/2)*RT22
[UU,SS] = svd(minsqrtCY*RT22); 

if n == 0 % Offer possibility to choose model order
    trap = db(diag(SS)) - min(db(diag(SS)));
    clf; bar(trap); hold on;
    [maxsstep,ind] = max(abs(diff(trap)));
    bar(1:ind,trap(1:ind),'r')
    ylabel('Singular Values [dB]');
    xlabel('Model Order');
    disp(['Max step at order ' num2str(ind) ', ' num2str(maxsstep) ' dB.'])
    shg;
    n = input('Please enter model order: ');
end

% 1.i. Or=U(:,1:n)
Or = sqrtCY*UU(:,1:n); % Extended observability matrix

% 2. Estimate A and C with Or
A = pinv(Or(1:p*r-p,:))*Or(p+1:r*p,:); % Use shift property Or
C = Or(1:p,:);

% 3. Estimate B and D given A,C and H: (W)LS estimate
    % Compute sqrt(covarH^-1)
c = zeros(size(covarH));
for f = 1:F
    c(:,:,f) = fSqrtInverse(covarH(:,:,f));
end
    % Compute partial derivative of sqrt(covarH^-1)*(H_est - H) w.r.t. B (H_est(f) = C*inv(z(f)*I - A)*B + D)
[~,JB] = fJacobFreqSS(A,zeros(n,m),C,z); % Matrix form, unweighted
JB = reshape(permute(JB,[1 2 4 3]),[p*m F m*n]); % Vectorized form, unweighted
JB = fWeightJacobSubSpace(JB,c,p*m,F,m*n); % Add weighting
JB = reshape(JB,p*m*F,m*n);
    % Compute partial derivative of sqrt(covarH^-1)*(H_est - H) w.r.t. D
JD = zeros(p*m*F,p*m);
for k = 1:F
    JD(p*m*(k-1)+1:k*p*m,:) = c(:,:,k);
end
    % Compute -sqrt(covarH^-1)*(0 - H), i.e. minus the weighted error when
    % considering zero intial estimates for H_est
H2 = reshape(fWeightJacobSubSpace(reshape(H,p*m,F),c,p*m,F,1),[p*m*F,1]);
    
H_re = [real(H2); imag(H2)]; % Separate real and imaginary part weighted error
J = [JB JD]; % B and D as one parameter vector => concatenate Jacobians
J_re = [real(J); imag(J)]; % Separate real and imaginary part Jacobian
[J_re,schaling] = fNormalizeColumns(J_re); % Normalize columns of Jacobian with their rms value
    % Compute Gauss-Newton parameter update
[u,s,v] = svd(J_re,0);
temp = v*diag(1./diag(s))*u.'*H_re;
temp = temp./schaling';
    % Parameter estimates = parameter update, since zero initial estimates considered
B = zeros(n,m);
B(:) = temp(1:n*m);
D = zeros(p,m);
D(1:p*m) = temp(n*m+1:end);

    % Check stability of the estimated model
if fIsUnstable(A,'z')
    disp('Unstable model!!!')
    unstable = 1;
else
    unstable = 0;
end

    % Check number of output arguments and prepare output
if nargout == 4
    varargout{1} = A;
    varargout{2} = B;
    varargout{3} = C;
    varargout{4} = D;
elseif nargout == 5
    varargout{1} = A;
    varargout{2} = B;
    varargout{3} = C;
    varargout{4} = D;
    varargout{5} = unstable;
else
    error('Wrong number of output parameters!')
end

%}