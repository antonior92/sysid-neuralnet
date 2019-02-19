function [Jn,scaling] = fNormalizeColumns(J)
%FNORMALIZECOLUMNS Normalizes the columns of a matrix with their rms value.
%
%	Usage:
%		[Jn,scaling] = fNormalizeColumns(J)
%
%	Description:
%		Normalizes the columns of a matrix (e.g. a Jacobian) to have unit
%		rms value. Zero columns are unchanged, and get assigned unit
%		scaling.
%
%	Output parameters:
%		Jn : matrix with normalized columns
%       scaling : rms values of the columns of J
%
%	Input parameters:
%		J : matrix with unnormalized columns
%
%	Example:
%       % If J is a regression matrix, its columns can be scaled to
%       % possibly obtain a better conditioned least-squares problem. The
%       % scaling factors can then be used to compute the unnormalized
%       % parameter values.
%       N = 100; % Number of data samples
%       J = [9e14*randn(N,1) randn(N,1)]; % Badly conditioned regression matrix
%       theta = [2; 3]; % Parameter vector
%       y = J*theta; % Output vector
%       [Jn,scaling] = fNormalizeColumns(J); % Normalize columns J
%       theta_n = (Jn\y)./scaling(:); % Estimated parameter vector from normalized regressors
%       theta_u = J\y; % Estimated parameter vector from unnormalized regressors
%
%	Versions:
%		1.0 : August 31, 2015
%       1.1 : April 20, 2016
%           Help updated
%
%	Copyright (c) Vrije Universiteit Brussel – dept. ELEC
%   All rights reserved.
%   Software can be used freely for non-commercial applications only.
%   Disclaimer: This software is provided “as is” without any warranty.

%--------------------------------------------------------------------------
% Version 1.0 = Version 1.1
%--------------------------------------------------------------------------
% {

scaling = rms(J); % Rms values of each column
scaling(scaling == 0) = 1; % Robustify against columns with zero rms value
Jn = J*diag(1./scaling.'); % Scale columns with 1/rms value

%}