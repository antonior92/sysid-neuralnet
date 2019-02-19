function out = fOne(p,m,i)
%FONE Constructs a matrix with only one one, and zeros elsewhere.
%
%	Usage:
%		out = fOne(p,m,i)
%
%	Description:
%		Constructs a p x m matrix with a one in position i, and zeros
%		elsewhere.
%       This is useful in constructing the Jacobians of the error
%		e(f) = G_hat(f) - G(f) w.r.t. the elements of a state-space matrix
%		(A, B, C, or D, but typically D, since the Jacobian w.r.t. the
%		elements of D is a zero matrix where one element is one), where
%		G_hat(f) = C*(z(f)*I - A)^(-1)*B + D. 
%
%	Output parameters:
%		out : p x m matrix with a one in position i (out(i) = 1), and
%		      zeros elsewhere
%
%	Input parameters:
%		p : number of rows in out
%       m : number of columns in out
%       i : position at which out has a one (out(i) = 1, or out(k,l) = 1,
%           where i = sub2ind([p m],k,l))
%
%`  Example:
%       out = fOne(2,3,3); % A 2 x 3 zero matrix with a one at position 3
%       % => out = [0 1 0;
%       %           0 0 0];
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
%
%	See also sub2ind

%--------------------------------------------------------------------------
% Version 1.0 = Version 1.1
%--------------------------------------------------------------------------
% {

out = zeros(p,m);
out(i) = 1;

%}