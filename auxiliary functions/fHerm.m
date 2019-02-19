function B = fHerm(A)
%FHERM Average of square matrix and its Hermitian transpose.
%
%	Usage:
%		B = fHerm(A)
%
%	Description:
%		Computes the average of a square matrix A and its complex conjugate
%		transpose A'.
%
%	Output parameters:
%		B : average of A and A'
%
%	Input parameters:
%		A : square matrix
%
%	Versions:
%		1.0 : September 2, 2015
%       1.1 : April 19, 2016
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

B = (A + A')/2;

%}