function out = fVec(in)
%FVEC Vectorization of a matrix or tensor.
%
%	Usage:
%		out = fVec(in)
%
%	Description:
%		Stacks all the columns of a matrix/tensor on top of each other.
%
%	Output parameters:
%		out : numel(in) x 1 vector with all columns of in stacked on top of
%		      each other
%
%	Input parameters:
%		in : matrix or tensor
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

out = in(:);

%}