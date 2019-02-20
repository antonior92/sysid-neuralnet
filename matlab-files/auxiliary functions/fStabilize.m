function [A,B,C,D] = fStabilize(A,B,C,D,domain,highest_s)
%FSTABILIZE Stabilize a linear state-space model.
%
%   Deprecated!
%
%	Usage:
%		[A,B,C,D] = fStabilize(A,B,C,D,'s',highest_s)
%       [A,B,C,D] = fStabilize(A,B,C,D,'z',[])
%
%	Description:
%		Stabilizes a state-space model by reflecting the unstable poles
%		w.r.t. the stability border (i.e. in discrete-time, unstable poles
%		are mirrored w.r.t the unit circle; in continuous-time, unstable
%		poles are mirrored w.r.t. the imaginary axis). Like this, the
%		amplitude characteristic of the unstable part remains the same, but
%		the phase is changed. An all-pass section to compensate for this
%		phase change is not added, as it increases the model order.
%       => Take care when using this function that the phase of the
%       stabilized FRF is not the same as the phase of the non-stabilized
%       FRF.
%       After reflecting the unstable poles, the FRFs of the stable and the
%       stabilized unstable part are summed, and a state-space model is
%       estimated on the sum of these FRFs. Note that this new estimate is
%       not guaranteed to be stable.
%
%	Output parameters:
%		A : n x n stabilized state matrix
%       B : n x m stabilized input matrix
%       C : p x n stabilized output matrix
%       D : p x m stabilized feed-through matrix
%
%	Input parameters:
%		A : n x n state matrix
%       B : n x m input matrix
%       C : p x n output matrix
%       D : p x m feed-through matrix
%       domain : 's' for continuous-time state-space models
%                'z' for discrete-time state-space models
%       highest_s : largest s-value up to which the FRF of the stable and
%                   the stabilized unstable part are calculated before
%                   estimating a new state-space model on top of the sum of
%                   both
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
%
%   See also fss2frf, fFreqDomSubSpace, fss2frfCT, fFreqDomSubSpaceCT

%--------------------------------------------------------------------------
% Version 1.0 = Version 1.1
%--------------------------------------------------------------------------
% {

n = length(A); % State-space order
[T,AT] = eig(A); % State transformation to diagonalize state matrix and diagonalized state matrix
lambda = diag(AT); % Poles of the system
CT = C*T; % Transformed output matrix
BT = T^-1*B; % Transformed input matrix

% Compute indices of the states corresponding to stable and unstable poles
switch domain
    case 'z'
        index_unstable = find(abs(lambda) > 1);
        index_stable = find(abs(lambda) <= 1);
    case 's'
        index_unstable = find(real(lambda) > 0);
        index_stable = find(real(lambda) <= 0);
    otherwise
        error('domain unknown')
end

% Compute state-space realizations of the stable and unstable part
T = eye(n);
Tu = T(:,index_unstable); % Transformation that selects states corresponding to unstable poles
Ts = T(:,index_stable); % Transformation that selects states corresponding to stable poles
if ~isempty(Ts)
    As = Ts'*AT*Ts;
    Bs = Ts\BT;
    Cs = CT*Ts;
else
    As = 0;
    Bs = 0;
    Cs = 0;
end
if ~isempty(Tu)
    Au = Tu'*AT*Tu;
    Bu = Tu\BT;
    Cu = CT*Tu;
else
    Au = 0;
    Bu = 0;
    Cu = 0;
end

switch domain
    case 'z'
        freq = 0:0.001:0.499; % Frequency axis used to estimate FRFs
        Gs = fss2frf(As,Bs,Cs,D,freq); % FRF stable part
        Gstab = Gs + lfFlippole(Au,Bu,Cu,D,freq); % Gstab = FRF stable part + FRF stabilized part
        [A,B,C,D] = fFreqDomSubSpace(Gstab,0,freq,n,n+1); % Estimate state-space model on Gstab
    case 's'
        s = 1j*linspace(0,abs(highest_s),1000); % s-axis used to estimate FRFs
        Gs = fss2frfCT(As,Bs,Cs,D,s); % FRF stable part
        Gstab = Gs + lfFlippoleCT(Au,Bu,Cu,D,s); % Gstab = FRF stable part + FRF stabilized part
        if any(isnan(Gstab)) % NaNs in Gstab possible if e.g. A = C = 0, and s = 0 => C*(sI - A)^-1 = 0*Inf = NaN
            [~,~,indNaN] = ind2sub(size(Gstab),find(isnan(Gstab)));
            Gstab(:,:,indNaN) = []; % Remove NaNs from Gstab
            s(indNaN) = []; % Remove corresponding s-values
        end
        [A,B,C,D] = fFreqDomSubSpaceCT(Gstab,0,s,n,n+1); % Estimate state-space model on Gstab
end
    warning('Using fStabilize is deprecated.')
    % FIXME: Final state-space model is re-estimated from the FRM of a
    %        stabilized state-space model => no guarantee of stability
    % NOTE: Unstable poles reflected => amplitude characteristic the same,
    %       but phase changed. No all-pass section added to compensate for
    %       phase change, since model order would increase.
end

function Gflip = lfFlippole(Au,Bu,Cu,D,freq)
%LFFLIPPOLE Calculate FRF of discrete-time state-space model where poles are mirrored w.r.t. the unit circle and direct feed-through is removed.
    z = exp(1j*2*pi*freq);
    Gflip = zeros([size(D) length(freq)]);
    n = length(Au);
    Au = Au^-1;
    for f = 1:length(freq)
        Gflip(:,:,f) = Cu*(z(f)*eye(n)-Au)^-1*Bu;
    end
end

function Gflip = lfFlippoleCT(Au,Bu,Cu,D,s)
%LFFLIPPOLECT Calculate FRF of continuous-time state-space model where poles are mirrored w.r.t. the imaginary axis and direct feed-through is removed.
    Gflip = zeros([size(D) length(s)]);
    n = length(Au);
    Au = -real(Au) + 1j*imag(Au);
    for f = 1:length(s)
        Gflip(:,:,f) = Cu*(s(f)*eye(n)-Au)^-1*Bu;
    end
end

%}