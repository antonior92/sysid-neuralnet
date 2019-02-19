close all
clear
clc

%% Add paths

addpath('./data/SilverboxFiles')
addpath('./auxiliary functions')

%% Load the Silverbox benchmark data

% Load the data
load SNLS80mV.mat V1 V2;    % V1 is the input, V2 is the output

% Sampling frequency
fs = 1e7/2^14;              % Sampling frequency = ca. 610 Hz

% The first part in the data is from a Gaussian input with increasing rms
% value. This part of the data is used as test data.
Ntest = 40700;   % Number of samples in test set

% The second part in the data is from 10 multisine realizations with
% 8192 samples per realization and 500 transient samples before each
% realization. The first 9 of these realizations are used as estimation
% data. The last realization is used as validation data (to do e.g. model
% order selection).
lines = 2:2:2680;   % Excited frequency lines (odd multisine)
Ntrans = 500;       % Number of transient samples before each realization
N = 8192;           % Number of samples per realization
R = 9;              % Number of multisine realizations in estimation set
Rval = 10 - R;      % Number of multisine realizations in validation set

% Extract estimation data
u = zeros(N,R); % Preallocate
y = zeros(N,R); % Preallocate
for r = 0:R-1   % Loop over realizations
    Nstart = Ntest + r*(N+Ntrans);  % Starting sample realization r+1
    Nstop = N + Nstart - 1;         % Ending sample realization r+1
    u(:,r+1) = V1(Nstart:Nstop).';  % Extract input realization r+1
    y(:,r+1) = V2(Nstart:Nstop).';  % Extract output realization r+1
end

% Extract validation data
uval = zeros(N,Rval);	% Preallocate
yval = zeros(N,Rval);	% Preallocate
for r = R:10-1          % Loop over realizations
    Nstart = Ntest + r*(N+Ntrans);      % Starting sample realization r+1
    Nstop = N + Nstart - 1;             % Ending sample realization r+1
    uval(:,r-R+1) = V1(Nstart:Nstop).';	% Extract input realization r+1
    yval(:,r-R+1) = V2(Nstart:Nstop).';	% Extract output realization r+1
end

% Extract test data
utest = V1(1:Ntest).';	% Input test data
ytest = V2(1:Ntest).';	% Output test data

%% Estimate nonparametric linear model (BLA)

u = permute(u,[1,3,2,4]); % N x m x R x P estimation input
y = permute(y,[1,3,2,4]); % N x p x R x P estimation output

U = fft(u); U = U(lines,:,:,:); % Input spectrum at excited lines
Y = fft(y); Y = Y(lines,:,:,:); % Output spectrum at excited lines

% Estimate best linear approximation, total distortion, and noise distortion
[G,covGtotal,covGnoise] = fCovarFrf(U,Y);	% G: FRF; covGtotal: noise + NL; covGnoise: noise (all only on excited lines)

%% Estimate linear state-space model (frequency domain subspace)

% Choose model order
na = 2;                 % Model order (only a single model order in this demo)
maxr = 10;              % Subspace dimensioning parameter
freq = (lines-1)*fs/N;	% Excited frequencies (in Hz)

% covGtotal = repmat(eye(1),[1 1 length(lines)]);                         % Uncomment for uniform weighting (= no weighting)
maxIter = 100;                                                          % Number of Levenberg-Marquardt iterations
models = fLoopSubSpace(freq,G,covGtotal,na,maxr,maxIter,true,true,fs);	% Estimated subspace model

% Extract state-space matrices
model = models{na};         % Select the model of order na
[A,B,C,D] = model{:};       % Extract the state-space matrices of that model
[A,B,C] = dbalreal(A,B,C);  % Balanced realization

%% Plot FRF and impulse response linear model

figure
plot((lines-1)*fs/N,db(squeeze(G)),'.')
xlabel('Frequency (Hz)')
ylabel('Amplitude (dB)')
title('Frequency response function linear model')

figure
plot((lines-1)*fs/N,180/pi*phase(squeeze(G)),'.')
xlabel('Frequency (Hz)')
ylabel('Phase (degrees)')
title('Frequency response function linear model')

[b,a] = tfdata(ss(A,B,C,D,1/fs),'v');
figure
plot(filter(b,a,[1; zeros(N-1,1)]),'r')
xlabel('Sample number')
ylabel('Amplitude')
title('Impulse response linear model')
xlim([0 250])