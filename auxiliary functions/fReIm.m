function B = fReIm(A)
%FREIM Stacks the real and imaginary part of a matrix on top of each other.
%
%	Usage:
%		B = fReIm(A)
%
%	Description:
%		Stacks the real and imaginary part of matrix A on top of each
%		other, i.e. B = [real(A); imag(A)].
%       To stack the real and imaginary part next to each other, i.e.
%       [real(A) imag(A)], use B = fReIm(A.').'
%
%	Output parameters:
%		B : real matrix with real and imaginary part of A stacked on top of
%		    each other
%
%	Input parameters:
%		A : complex matrix
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

%--------------------------------------------------------------------------
% Version 1.0 = Version 1.1
%--------------------------------------------------------------------------
% {

B = [real(A); imag(A)];

%}