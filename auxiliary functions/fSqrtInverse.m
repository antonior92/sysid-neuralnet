function B = fSqrtInverse(A)
%FSQRTINVERSE Computes B, such that B*B = inv(A).
%
%	Usage:
%		B = fSqrtInverse(A)
%
%	Description:
%		B = fSqrtInverse(A) computes the inverse of the matrix square root
%		of a square positive definite matrix A.
%
%	Output parameters:
%		B : inverse of matrix square root of B
%
%	Input parameters:
%		A : positive definite matrix
%
%   Example:
%       V = orth(randn(3,3)); % Random eigenvectors
%       D = diag(1+abs(randn(3,1))); % Random positive eigenvalues
%       A = V*D*V.'; % Positive definite random square matrix
%       B = fSqrtInverse(A);
%       disp('B*B'), B*B
%       disp('inv(A)'), inv(A)
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
% Version 1.0 = Version 1.1
%--------------------------------------------------------------------------
% {

if ~(ismatrix(A) && (size(A,1) == size(A,2)))
    error('A should be a square matrix')
end

[U,S] = svd(A,0);
B = U*diag(diag(S).^(-1/2))*U';

%}