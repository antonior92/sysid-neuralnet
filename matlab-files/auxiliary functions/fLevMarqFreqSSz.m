function [A,B,C,D] = fLevMarqFreqSSz(freq,G,covG,A0,B0,C0,D0,MaxCount)
%FLEVMARQFREQSSZ Optimize state-space matrices using Levenberg-Marquardt.
%
%	Usage:
%		[A,B,C,D] = fLevMarqFreqSSz(freq,G,covG,A0,B0,C0,D0,MaxCount)
%
%	Description:
%		Optimize state-space matrices A, B, C, and D using at most MaxCount
%		Levenberg-Marquardt runs starting from initial values A0, B0, C0,
%		and D0. The cost function that is optimized is the weighted sum of
%		the mean-square differences between the provided frequency response
%		matrix G at the normalized frequencies freq and the estimated
%		frequency response matrix G_hat = C*(z(freq)*I - A)^(-1)*B + D.
%       The cost function is weighted with the inverse of the covariance
%       matrix of G, if this covariance matrix covG is provided (no
%       weighting if covG = 0).
%
%	Output parameters:
%		A : n x n optimized state matrix
%       B : n x m optimized input matrix
%       C : p x n optimized output matrix
%       D : p x m optimized feed-through matrix
%
%	Input parameters:
%       freq : vector of normalized frequencies at which the FRM is given
%              (0 < freq < 0.5)
%		G : p x m x F array of FRF (or FRM) data
%		covG : p*m x p*m x F covariance tensor on G
%              (0 if no weighting required)
%		A0 : n x n initial state matrix
%       B0 : n x m initial input matrix
%       C0 : p x n initial output matrix
%       D0 : p x m initial feed-through matrix
%		MaxCount : maximum number of Levenberg-Marquardt optimizations
%                  (maximum 1000, if MaxCount = Inf, then the algorithm
%                  stops if the cost functions of the last 10 successful
%                  steps differ by less than 0.1% or if 1000 iterations
%                  were performed)
%
%	Reference:
%       Pintelon, R. and Schoukens, J. (2012). System Identification: A
%       frequency domain approach. Wiley-IEEE Press, second edition.
%
%	Versions:
%		1.0 : August 28, 2015
%       1.1 : April 20, 2016
%           Help updated
%
%	Copyright (c) Vrije Universiteit Brussel – dept. ELEC
%   All rights reserved.
%   Software can be used freely for non-commercial applications only.
%   Disclaimer: This software is provided “as is” without any warranty.
%
%	See also fss2frf, fSqrtInverse, fVec, fReIm, fOne, fJacobFreqSS, fWeightJacobSubSpace, fNormalizeColumns

%--------------------------------------------------------------------------
% Version 1.0 = Version 1.1
%--------------------------------------------------------------------------
% {

[A,B,C,D] = deal(A0,B0,C0,D0); % Parameter values from initial values
n = length(A); % Model order
[p,m] = size(D); % Number of outputs/inputs
npar = n^2 + n*m + p*n + p*m; % Number of parameters
z = exp(1j*2*pi*freq(:)); % z vector
info = false; % Display iteration number and cost function or not
GSS_model = fss2frf(A,B,C,D,freq); % Modeled frequency response matrix
F = length(freq); % Number of frequencies

% Compute weighting
if covG == 0 % No weighting
    c = repmat(eye(p*m),[1 1 F]);
else % Weight error with the square root of the inverse of the covariance matrix of G
    c = zeros(size(covG));
    for f = 1:F
        c(:,:,f) = fSqrtInverse(covG(:,:,f));
    end
end

% Compute weighted error
err_old = zeros(F*p*m,1); % Preallocating
for i = 1:F
    err_old((i-1)*p*m+1:i*p*m) = c(:,:,i)*(fVec(GSS_model(:,:,i) - G(:,:,i)));
end
err_old = fReIm(err_old); % Stack real and imaginary part on top of each other

% Compute cost function (= sum of squares of weighted errors)
K_old = err_old'*err_old;

% Initialization of the Levenberg-Marquardt loop
Count = 0; % Iteration number
lambda = -1; % Damping factor
Kmemory = zeros(10,1); % Store cost functions of last ten iterations
JD = zeros(p,m,p*m); % Jacobian of weighted error w.r.t. parameter values feed-through matrix D
for i = 1:p*m
    JD(:,:,i) = fOne(p,m,i);
end
JD = repmat(JD,[1 1 1 F]);

while Count < MaxCount

    % Compute Jacobian of weighted error w.r.t. all parameter values
    [JA,JB,JC] = fJacobFreqSS(A,B,C,z); % Jacobians w.r.t. parameter values A, B, and C matrices
    temp = zeros(p,m,npar,F);
    temp(:,:,                  1:n^2,:) = JA;
    temp(:,:,            n^2 + (1:n*m),:) = JB;
    temp(:,:,      n*m + n^2 + (1:n*p),:) = JC;
    temp(:,:,n*p + n*m + n^2 + 1:end,:) = JD;
    temp = permute(temp,[1 2 4 3]);
    if covG ~= 0
        temp = reshape(temp,[p*m F npar]); % Unweighted Jacobian
        temp = fWeightJacobSubSpace(temp,c,p*m,F,npar); % Add weighting
    end
    temp = reshape(temp,F*p*m,npar);
    J = fReIm(temp); % Stack real and imaginary part on top of each other

    % For testing purposes:
    %{
        Jnum = fReIm(lfJacobNum(A,B,C,D,freq,n,m,p)); % Numerically calculated Jacobian (unweighted)
        figure
        for i = 1:size(J,2)
            plot(J(:,i) - Jnum(:,i)); % Compare analytical and numerical Jacobians
            title(num2str(i));
            pause(1);
        end
    %}

    [J,scaling] = fNormalizeColumns(J); % Normalize columns of Jacobian

    K = K_old;
    [U,S,V] = svd(J,0); % SVD of Jacobian (to efficiently compute inverse of Jacobian)

    % First Marquardt loop: initialize lambda (see Pintelon and Schoukens, 2012)
    if lambda == -1 % Initial damping factor
        lambda = S(1,1); % Put initial damping factor equal to the largest singular value of the initial Jacobian
    end

    while K >= K_old && Count < MaxCount % As long as the step is unsuccessful

        % Determine the rank of the Jacobian
        s = diag(S); % Singular values
        tol = max(size(J))*eps(max(s)); % Tolerance
        r = sum(s > tol); % Number of non-zero (within tolerance) singular values (=> rank estimate)

        % Calculate the parameter step
        s = zeros(r,r);
        for k = 1:r
            s(k,k) = S(k,k)/(S(k,k)^2 + lambda^2);
        end
        dq = -V(:,1:r)*s*U(:,1:r).'*err_old; % Parameter step
        dq = dq./scaling'; % Compensate for normalization

        % Compute the frequency response matrix with the new parameter values
        [dA,dB,dC,dD] = lfVec2Par(dq,n,m,p);
        Atest = A + dA;
        Btest = B + dB;
        Ctest = C + dC;
        Dtest = D + dD;
        GSS_model_test = fss2frf(Atest,Btest,Ctest,Dtest,freq);

        % Compute the weighted error
        if covG == 0 % No weighting
            err = fReIm(GSS_model_test(:) - G(:));
        else % Weighting with the square root of the inverse of the covariance matrix of G
            err = reshape(fWeightJacobSubSpace(reshape(GSS_model_test - G,p*m,F),c,p*m,F,1),[p*m*F,1]);
            err = fReIm(err(:)); % Stack real and imaginary part on top of each other
        end
        
        % Compute the new cost function
        K = err'*err;

        % Check successfulness of the step
        if K >= K_old % Step unsuccessful -> increase lambda
            lambda = lambda*sqrt(10);
        else          % Step successful   -> decrease lambda
            lambda = lambda/2;
            if info == 1 % Display iteration number and cost function if requested
                disp([num2str(Count) '  ' num2str(K,6)]);
            end
            if isinf(MaxCount) % If MaxCount = Inf was specified
                Kmemory = [K; Kmemory(1:end-1)]; % Store cost functions of last 10 successful steps
                if abs((K-Kmemory(end)))/K < 1e-3 % Stop if cost functions last 10 successful steps differ by less than 0.1%
                    Count = Inf;
                end
            end
        end
        Count = Count + 1; % Increase iteration number
        if Count == 1000 % Abort the algorithm after 1000 iterations
            Count = Inf;
            disp('Aborted at Count=1000')
        end
    end

    % Adopt the new parameter values, cost function and error if the step was successful
    if K < K_old
        K_old = K;
        A = Atest;
        B = Btest;
        C = Ctest;
        D = Dtest;
        err_old = err;
    end
end
end

% For testing purposes
%{
function J = lfJacobNum(A,B,C,D,freq,n,m,p)
%LFJACOBNUM Numerically compute the Jacobian of the (unweighted) error
q0 = lfPar2Vec(A,B,C,D);
J = zeros(length(freq)*p*m,length(q0));
epsilon = 1e-5;
for i = 1:length(q0)
    qmin = q0;
    qplus = q0;
    qmin(i) = qmin(i)*(1 - epsilon);
    qplus(i) = qplus(i)*(1 + epsilon);
    [Amin,Bmin,Cmin,Dmin] = lfVec2Par(qmin,n,m,p);
    [Aplus,Bplus,Cplus,Dplus] = lfVec2Par(qplus,n,m,p);
    Gmin = fss2frf(Amin,Bmin,Cmin,Dmin,freq);
    Gplus = fss2frf(Aplus,Bplus,Cplus,Dplus,freq);
    Jtemp = (Gplus - Gmin)/(2*epsilon*q0(i));
    J(:,i) = Jtemp(:);
end
end

function q = lfPar2Vec(A,B,C,D)
%LFPAR2VEC Construct parameter vector from state-space matrices
q = [A(:); B(:); C(:); D(:)];
end
%}

function [A,B,C,D] = lfVec2Par(q,n,m,p)
%LFVEC2PAR Construct state-space matrices from parameter vector
A = zeros(n); % Preallocate
B = zeros(n,m); % Preallocate
C = zeros(p,n); % Preallocate
D = zeros(p,m); % Preallocate

A(:) = q(                  1:n^2); % Extract first n*n parameters from q and assign them to matrix A
B(:) = q(            n^2 + (1:n*m)); % Extract next n*m parameters from q and assign them to matrix B
C(:) = q(      n*m + n^2 + (1:p*n)); % Extract next p*n parameters from q and assign them to matrix C
D(:) = q(p*n + n*m + n^2 + 1:end); % Extract final p*m parameters from q and assign them to matrix D
end

%}