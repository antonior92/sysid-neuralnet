function unstable = fIsUnstable(A,domain)
%FISUNSTABLE Determines if a linear state-space model is unstable.
%
%	Usage:
%		unstable = fIsUnstable(A,'s')
%       unstable = fIsUnstable(A,'z')
%
%	Description:
%       unstable = fIsUnstable(A,'s') determines if a continuous-time
%       state-space model with state matrix A is unstable.
%		unstable = fIsUnstable(A,'z') determines if a discrete-time
%		state-space model with state matrix A is unstable.
%
%	Output parameters:
%		unstable : boolean indicating whether or not the state-space model
%                  is unstable
%
%	Input parameters:
%		A : n x n state matrix
%       domain : 's' for continuous-time state-space models
%                'z' for discrete-time state-space models
%
%	Versions:
%		1.0 : August 27, 2015
%       1.1 : April 19, 2016
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

unstable = false;
switch domain
    case 'z' % In discrete-time
        if any(abs(eig(A)) > 1) % Unstable if at least one pole outside unit circle
            unstable = true;
        end
    case 's' % In continuous-time
        if any(real(eig(A)) > 0) % Unstable if at least one pole in right-half plane
            unstable = true;
        end
    otherwise
        error('domain unknown')
end

%}