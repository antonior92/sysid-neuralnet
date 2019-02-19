function GSS = fss2frfCT(A,B,C,D,s)
%FSS2FRFCT Compute frequency response function from state-space parameters (continuous-time).
%
%	Usage:
%		GSS = fss2frfCT(A,B,C,D,s)
%
%	Description:
%		GSS = fss2frfCT(A,B,C,D,s) computes the frequency response function
%		(FRF) or matrix (FRM) GSS at the frequencies s/(2*pi*1j) from the
%		state-space matrices A, B, C, and D. 
%       GSS(f) = C*inv(1j*2*pi*f*I - A)*B + D
%
%	Output parameters:
%		GSS : p x m x F frequency response matrix
%
%	Input parameters:
%		A : n x n state matrix
%       B : n x m input matrix
%       C : p x n output matrix
%       D : p x m feed-through matrix
%       s : vector s = 1j*2*pi*freq, where freq is a vector of frequencies
%           (in Hz) at which the FRM is computed
%
%	Versions:
%		1.0 : September 1, 2015
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

GSS = zeros([size(D) length(s)]); % Preallocate
I = eye(size(A,1)); % n x n identity matrix
for f = 1:length(s) % Loop over all frequencies
%     GSS(:,:,f) = C*(s(f)*I - A)^-1*B + D;
    GSS(:,:,f) = C*((s(f)*I - A)\B) + D; % NOTE: faster, but slightly different result
end

%}