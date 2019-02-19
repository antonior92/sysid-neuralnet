function GSS = fss2frf(A,B,C,D,freq)
%FSS2FRF Compute frequency response function from state-space parameters (discrete-time).
%
%	Usage:
%		GSS = fss2frf(A,B,C,D,freq)
%
%	Description:
%		GSS = fss2frf(A,B,C,D,freq) computes the frequency response
%		function (FRF) or matrix (FRM) GSS at the normalized frequencies
%		freq from the state-space matrices A, B, C, and D.
%       GSS(f) = C*inv(exp(1j*2*pi*f)*I - A)*B + D
%
%	Output parameters:
%		GSS : p x m x F frequency response matrix
%
%	Input parameters:
%		A : n x n state matrix
%       B : n x m input matrix
%       C : p x n output matrix
%       D : p x m feed-through matrix
%       freq : vector of normalized frequencies at which the FRM is
%              computed (0 < freq < 0.5)
%
%	Versions:
%		1.0 : August 27, 2015
%       1.1 : April 20, 2016
%           Help updated
%
%	Copyright (c) Vrije Universiteit Brussel – dept. ELEC
%   All rights reserved.
%   Software can be used freely for non-commercial applications only.
%   Disclaimer: This software is provided “as is” without any warranty.

%--------------------------------------------------------------------------
% Version 1.0
%--------------------------------------------------------------------------
% {

z = exp(1j*2*pi*freq(:)); % Z-transform variable
GSS = zeros([size(D) length(freq)]); % Preallocate
I = eye(size(A,1)); % n x n identity matrix
for f = 1:length(freq) % Loop over all frequencies
%     GSS(:,:,f) = C*(z(f)*I - A)^-1*B + D;
    GSS(:,:,f) = C*((z(f)*I - A)\B) + D; % NOTE: faster, but slightly different result
end

%}