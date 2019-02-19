function out = fWeightJacobSubSpace(Jacob,W,mp,F,npar)
%FWEIGHTJACOBSUBSPACE Adds weighting to an unweighted Jacobian.
%
%	Usage:
%		out = fWeightJacobSubSpace(Jacob,C,mp,F,npar)
%
%	Description:
%		Computes the Jacobian of the weighted error e_W(f) = W(:,:,f)*e(f)
%		w.r.t. the elements of a state-space matrix (A, B, C, or D), given
%		the Jacobian Jacob of the unweighted error e(f) = G_hat(f) - G(f)
%		w.r.t. the elements of the same state-space matrix, where
%		G_hat(f) = C*(z(f)*I - A)^(-1)*B + D.
%
%   Remark:
%       Can be used more generally to add weighting to an unweighted
%       Jacobian of a vector-valued function with mp outputs and npar
%       inputs, and sampled in F operating points.
%
%	Output parameters:
%		out : m*p x F x npar weighted Jacobian
%
%	Input parameters:
%		Jacob : m*p x F x npar unweighted Jacobian
%       W : m*p x m*p x F weighting matrix (e.g. square root of inverse
%           of covariance matrix of G)
%       mp : number of inputs times number of outputs (optional, determined
%            from Jacob if not provided)
%       F : number of frequencies at which the Jacobian is computed
%           (optional, determined from Jacob if not provided)
%       npar : number of parameters in the state-space matrix w.r.t. which
%              the Jacobian is taken (optional, determined from Jacob if
%              not provided)
%
%	Versions:
%		1.0 : September 1, 2015
%       1.1 : September 2, 2015
%           Faster implementation
%       1.2 : April 20, 2016
%           Help updated
%
%	Copyright (c) Vrije Universiteit Brussel – dept. ELEC
%   All rights reserved.
%   Software can be used freely for non-commercial applications only.
%   Disclaimer: This software is provided “as is” without any warranty.

%--------------------------------------------------------------------------
% Version 1.1 = Version 1.2
%   Faster implementation
%--------------------------------------------------------------------------
% {

% Determine number of parameters, frequencies, and/or inputs times outputs if not provided
if nargin < 5
    [s1,s2,s3] = size(Jacob);
    npar = s3; % Number of parameters
end
if nargin < 4
    F = s2; % Number of frequencies
end
if nargin < 3
    mp = s1; % Number of inputs times number of outputs
end

% Add weighting to the Jacobian
out = zeros(mp,npar,F);
Jacob_perm = permute(Jacob,[1 3 2]);
for i = 1:F
    out(:,:,i) = W(:,:,i)*Jacob_perm(:,:,i);
end
out = permute(out,[1 3 2]);

%}

%--------------------------------------------------------------------------
% Version 1.0
%--------------------------------------------------------------------------
%{

% Determine number of parameters, frequencies, and/or inputs and outputs
% if not provided
if nargin < 5
    [s1,s2,s3] = size(Jacob);
    npar = s3;
end
if nargin < 4
    F = s2;
end
if nargin < 3
    mp = s1;
end

% Add weighting to the Jacobian
out = zeros(mp,F,npar);
for i = 1:F
    for j = 1:npar
        out(:,i,j) = W(:,:,i)*Jacob(:,i,j);
    end
end

%}