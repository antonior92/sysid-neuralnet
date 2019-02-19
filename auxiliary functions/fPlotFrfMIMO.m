function fPlotFrfMIMO(G,freq,LineSpec,varargin)
%FPLOTFRFMIMO Make amplitude versus frequency plots of the elements of a frequency response matrix.
%
%	Usage:
%		fPlotFrfMIMO(G,freq)
%       fPlotFrfMIMO(G,freq,LineSpec)
%       fPlotFrfMIMO(G,freq,LineSpec,Name,Value,...)
%
%	Description:
%		fPlotFrfMIMO makes an amplitude versus frequency plot for each of
%		the components of the frequency response matrix G.
%
%   Remark:
%       fPlotFrfMIMO does not make a call to figure, so figure should be
%       called before fPlotFrfMIMO if you want to make sure that the plot
%       is made on a new figure.
%       fPlotFrfMIMO makes a call to hold on, so that consecutive calls of
%       fPlotFrfMIMO makes amplitude versus frequency plots of frequency
%       response matrices (with the same number of inputs and outputs) on
%       top of each other.
%
%	Input parameters:
%		G : p x m x F frequency response matrix (FRM)
%       freq : vector of frequencies at which the FRM is given (in Hz)
%       LineSpec : sets the line style, marker symbol, and color.
%                  (optional, default = 'b')
%       Name, Value pair arguments: see help plot for a list
%
%	Versions:
%		1.0 : August 31, 2015
%       1.1 : April 20, 2016
%           Help updated
%
%	Copyright (c) Vrije Universiteit Brussel – dept. ELEC
%   All rights reserved.
%   Software can be used freely for non-commercial applications only.
%   Disclaimer: This software is provided “as is” without any warranty.
%
%	See also fMetricPrefix, plot

%--------------------------------------------------------------------------
% Version 1.0 = Version 1.1
%--------------------------------------------------------------------------
% {

% Set default LineSpec if not provided
if nargin < 3
    LineSpec = 'b';
end

% Set scale of the frequency axis
[freqlabel,schaal] = fMetricPrefix(mean(freq));
freqlabel = ['Frequency [' freqlabel 'Hz]'];
freq = freq*schaal;

% Make the amplitude versus frequency plots
[p,m,~] = size(G);
for i = 1:p
    for j = 1:m
        subplot(p,m,sub2ind([m p],j,i))
        hold on
        plot(freq,db(squeeze(G(i,j,:))),LineSpec,varargin{:})
        if i == p % xlabels only on the bottom plots
            xlabel(freqlabel)
        end
        if j == 1 % ylabels only on the left plots
            ylabel('Amplitude [dB]')
        end
        title(['G_{' num2str(i) num2str(j) '}'])
    end
end

%}