function [JA,JB,JC] = fJacobFreqSS(A,B,C,z)
%FJACOBFREQSS Compute Jacobians of the unweighted errors w.r.t. elements A, B, and C matrices.
%
%	Usage:
%		[JA,JB,JC] = fJacobFreqSS(A,B,C,z)
%
%	Description:
%		Computes the Jacobians of the unweighted errors G_hat(f) - G(f)
%		w.r.t. the elements in the A, B, and C state-space matrices. Let
%		e(f) = G_hat(f) - G(f), where G_hat(f) = C*(z(f)*I - A)^(-1)*B + D
%		is the estimated and G(f) is the measured frequency response matrix
%		(FRM), then JA(:,:,sub2ind([n n],k,l),f) contains the partial
%		derivative of e(f) w.r.t. A(k,l), JB(:,:,sub2ind([n m],k,l),f)
%		contains the partial derivative of e(f) w.r.t. B(k,l), and
%       JC(:,:,sub2ind([p n],k,l),f) contains the partial derivative of
%       e(f) w.r.t. C(k,l).
%
%	Output parameters:
%		JA : p x m x n^2 x F tensor where JA(:,:,sub2ind([n n],k,l),f)
%		     contains the partial derivative of the unweighted error e(f)
%		     at frequency f w.r.t. A(k,l)
%		JB : p x m x n*m x F tensor where JB(:,:,sub2ind([n m],k,l),f)
%		     contains the partial derivative of e(f) w.r.t. B(k,l)
%		JC : p x m x p*n x F tensor where JC(:,:,sub2ind([p n],k,l),f)
%		     contains the partial derivative of e(f) w.r.t. C(k,l)
%
%	Input parameters:
%		A : n x n state matrix
%       B : n x m input matrix
%       C : p x n output matrix
%       z : F x 1 vector z = exp(1j*2*pi*freq(:)), where freq is a vector
%           of normalized frequencies at which the Jacobians are computed
%           (0 < freq < 0.5)
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
%	See also fLevMarqFreqSSz, sub2ind, ind2sub, fOne

%--------------------------------------------------------------------------
% Version 1.0 = Version 1.1
%--------------------------------------------------------------------------
% {

F = length(z); % Number of frequencies
n = size(A,1); % Number of states
m = size(B,2); % Number of inputs
p = size(C,1); % Number of outputs

JA = NaN(p,m,n*n,F); % Preallocate
JB = NaN(p,m,n*m,F); % Preallocate
JC = NaN(p,m,n*p,F); % Preallocate

[k,ell] = ind2sub([n n],1:n*n); % Rows and columns in A
I = eye(n); % n x n identity matrix
I_m = eye(m); % m x m identity matrix
I_p = eye(p); % p x p identity matrix
for f = 1:F % Loop over all frequencies
%     temp1 = (z(f)*I - A)^-1;
    temp1 = (z(f)*I - A)\I; % NOTE: faster, but slightly different result
    temp2 = C*temp1;
    temp3 = temp1*B;
    
    % Jacobian w.r.t. all elements in A
    % Note that the partial derivative of e(f) w.r.t. A(k(i),ell(i)) is equal to
    % temp2*fOne(n,n,i)*temp3, and thus JA(:,:,i,f) = temp2(:,k(i))*temp3(ell(i),:)
    for i = 1:n*n % Loop over all elements in A
        JA(:,:,i,f) = temp2(:,k(i))*temp3(ell(i),:); % Jacobian w.r.t. A(i)=A(k(i),ell(i))
    end
    
    % Jacobian w.r.t. all elements in B
    % Note that the partial derivative of e(f) w.r.t. B(k,l) is equal to
    % temp2*fOne(n,m,sub2ind([n m],k,l)), and thus JB(:,l,sub2ind([n m],k,l),f) = temp2(:,k)
    JB(:,:,:,f) = reshape(kron(I_m,temp2),p,m,m*n);
    
    % Jacobian w.r.t. all elements in C
    % Note that the partial derivative of e(f) w.r.t. C(k,l) is equal to
    % fOne(p,n,sub2ind([p n],k,l))*temp3, and thus JC(k,:,sub2ind([p n],k,l),f) = temp3(l,:)
    JC(:,:,:,f) = reshape(kron(temp3.',I_p),p,m,n*p);
end

%}