function [G,covGML,covGn] = fCovarFrf(U,Y)
%FCOVARFRF Computes frequency response matrix and noise and total covariance matrix from input/output spectra.
%
%	Usage:
%		[G,covGML,covGn] = fCovarFrf(U,Y)
%
%	Description:
%		Estimates the frequency response matrix, and the corresponding
%		noise and total covariance matrices from the spectra of periodic
%		input/output data.
%       covGn = 0 if only one period is specified.
%       covGML = 0 if only one experiment block of m experiments is
%       specified, i.e. there should be at least two times as many
%       experiments as inputs to be able to estimate the total covariance.
%
%	Output parameters:
%		G : p x m x F Frequency Response Matrix (FRM)
%       covGML : p*m x p*m x F total covariance (= stochastic nonlinear
%                contributions + noise) 
%       covGn : p*m x p*m x F noise covariance
%
%	Input parameters:
%		U : F x m x exp x P input spectra at F frequency lines, with m
%		    inputs, exp experiments (typically phase realizations of a
%		    multisine, exp is preferably an integer multiple of m), and P
%		    periods
%       Y : F x p x exp x P output spectra with p outputs
%
%   Example:
%       % SISO example (Hammerstein system)
%       f_NL = @(x) x + 0.2*x.^2 + 0.1*x.^3; % Nonlinear function
%       [b,a] = cheby1(2,10,2*0.1); % Filter coefficients
%       N = 1000; % Number of samples per period
%       Lines = 2:133; % Excited frequency lines;
%       R = 3; % Number of phase realizations
%       PTrans = 1; % Number of transient periods
%       P = 2; % Number of periods
%       u = fMultisine(N,Lines,[],R); % Input signal: three phase realizations of a random-phase multisine
%       u = u/mean(std(u)); % Scale input signal
%       u = repmat(u,[PTrans+P,1]); % One transient period + two periods
%       x = f_NL(u); % Intermediate signal
%       y0 = filter(b,a,x); % Noise-free output signal
%       y = y0 + 0.01*mean(std(y0))*randn(size(y0)); % Noisy output signal
%       u(1:PTrans*N,:) = []; % Remove transient period(s)
%       x(1:PTrans*N,:) = []; % Remove transient period(s)
%       y(1:PTrans*N,:) = []; % Remove transient period(s)
%       u = reshape(u,[N,P,R]); % Reshape input signal
%       y = reshape(y,[N,P,R]); % Reshape output signal
%       U = fft(u); % Input spectrum
%       Y = fft(y); % Output spectrum
%       U = U(Lines,:,:); % Select only excited frequency lines
%       Y = Y(Lines,:,:); % Select only excited frequency lines
%       U = permute(U,[1,4,3,2]); % F x m x R x P
%       Y = permute(Y,[1,4,3,2]); % F x p x R x P
%       [G,covGML,covGn] = fCovarFrf(U,Y); % Compute FRF, total and noise distortion
%       figure
%       plot(Lines,db([G(:) covGML(:) covGn(:)]))
%       xlabel('Frequency line')
%       ylabel('Amplitude (dB)')
%       legend('FRF','Total distortion','Noise distortion')
%
%	Versions:
%		1.0 : September 2, 2015
%       1.1 : April 18, 2016
%           Help updated
%
%	Copyright (c) Vrije Universiteit Brussel – dept. ELEC
%   All rights reserved.
%   Software can be used freely for non-commercial applications only.
%   Disclaimer: This software is provided “as is” without any warranty.
%
%	See also fMultisine, fVec, fHerm

%--------------------------------------------------------------------------
% Version 1.0 = Version 1.1
%--------------------------------------------------------------------------
% {

[F, m, exp, P] = size(U); % Number of frequencies, inputs, experiments, and periods
p = size(Y,2); % Number of outputs
M = floor(exp/m); % Number of blocks of experiments
if M*m ~= exp
    disp('Warning: suboptimal number of experiments: M*m ~= exp')
end
U = reshape(U(:,:,1:m*M,:),[F,m,m,M,P]); % Reshape in M blocks of m experiments
Y = reshape(Y(:,:,1:m*M,:),[F,p,m,M,P]); % Reshape in M blocks of m experiments

if P > 1
    % Average input/output spectra over periods
    U_mean = mean(U,5);
    Y_mean = mean(Y,5);

    % Estimate noise spectra
    NU = U - repmat(U_mean,[1,1,1,1,P]); % Estimated input noise spectra
    NY = Y - repmat(Y_mean,[1,1,1,1,P]); % Estimated output noise spectra
    NU = permute(NU,[2,3,4,5,1]); % m x m x M x P x F
    NY = permute(NY,[2,3,4,5,1]); % p x m x M x P x F

    % Calculate input/output noise (co)variances on averaged (over periods) spectra
    covU = zeros(m*m,m*m,F,M); % Preallocating
    covY = zeros(p*m,p*m,F,M); % Preallocating
    covYU = zeros(p*m,m*m,F,M); % Preallocating
    for mm = 1:M % Loop over experiment blocks
        NU_m = permute(NU(:,:,mm,:,:),[1 2 4 5 3]); % Input noise spectrum of experiment block mm (m x m x P x F)
        NY_m = permute(NY(:,:,mm,:,:),[1 2 4 5 3]); % Output noise spectrum of experiment block mm (p x m x P x F)
        for f = 1:F % Loop over frequencies
            tempuu = 0;
            tempyy = 0;
            tempyu = 0;
            for pp = 1:P % Loop over periods
                tempuu = tempuu + fVec(NU_m(:,:,pp,f))*(fVec(NU_m(:,:,pp,f)))';
                tempyy = tempyy + fVec(NY_m(:,:,pp,f))*(fVec(NY_m(:,:,pp,f)))';
                tempyu = tempyu + fVec(NY_m(:,:,pp,f))*(fVec(NU_m(:,:,pp,f)))';
            end
            covU(:,:,f,mm)  = tempuu/(P-1)/P; % Input noise covariance
            covY(:,:,f,mm)  = tempyy/(P-1)/P; % Output noise covariance
            covYU(:,:,f,mm) = tempyu/(P-1)/P; % Input/output noise covariance
        end
    end
    
    % Further calculations with averaged spectra
    U = U_mean;
    Y = Y_mean;
    clear U_mean Y_mean;
end

% Rearrange input/output spectra
U = permute(U,[2,3,4,1]); % m x m x M x F
Y = permute(Y,[2,3,4,1]); % p x m x M x F

% Compute FRM and noise and total covariance on averaged (over experiment blocks and periods) FRM
G = zeros(p,m,F); % Preallocating
covGML = zeros(m*p,m*p,F); % Preallocating
covGn  = zeros(m*p,m*p,F); % Preallocating
Gm = zeros(p,m,M); % Preallocating
U_inv_m = zeros(m,m,M); % Preallocating
covGn_m = zeros(m*p,m*p,M); % Preallocating
for f = 1:F
    % Estimate the frequency response matrix (FRM)
    for mm = 1:M
        [uu,ss,vv] = svd(U(:,:,mm,f)',0);
        U_inv_m(:,:,mm) = uu*diag(1./diag(ss))*vv';
        Gm(:,:,mm) = Y(:,:,mm,f)*U_inv_m(:,:,mm); % FRM of experiment block m at frequency f
    end
    G(:,:,f) = mean(Gm,3); % Average FRM over experiment blocks

    % Estimate the total covariance on averaged FRM
    temp = 0;
    NG = repmat(G(:,:,f),[1 1 M]) - Gm;
    for mm = 1:M
        temp = temp + fVec(NG(:,:,mm))*(fVec(NG(:,:,mm)))';
    end
    covGML(:,:,f) = temp/M/(M-1);

    % Estimate noise covariance on averaged FRM (only if P > 1)
    if P > 1
        for mm = 1:M
            U_invT = U_inv_m(:,:,mm).';
            A = kron(U_invT,eye(p));
            B = -kron(U_invT,Gm(:,:,mm));
            covGn_m(:,:,mm) = A*covY(:,:,f,mm)*A' + B*covU(:,:,f,mm)*B' + 2*fHerm(A*covYU(:,:,f,mm)*B');
        end
        covGn(:,:,f) = mean(covGn_m,3)/M;
    end
end

% No total covariance estimate possible if only one experiment block
if M < 2
    covGML = 0;
end

% No noise covariance estimate possible if only one period
if P == 1
    covGn = 0;
end

%}
