function [label,scale] = fMetricPrefix(in)
%FMETRICPREFIX Returns an appropriate metric prefix and a corresponding scaling factor.
%
%	Usage:
%		[label,scale] = fMetricPrefix(in)
%
%	Description:
%		Returns a metric prefix label and the appropriate scaling factor
%		with which in has to be multiplied in order to use the prefix.
%
%       The supported prefixes are:
%
%       prefix	label   scale   range
%       ------  -----   -----   -----
%               ''      1                   in <= 10^-16.5
%       femto   'f'     10^15   10^-16.5 <  in <= 10^-13.5
%       pico    'p'     10^12   10^-13.5 <  in <= 10^-10.5
%       nano    'n'     10^9    10^-10.5 <  in <= 10^-7.5
%       micro   '\mu'   10^6    10^-7.5  <  in <= 10^-4.5
%       milli   'm'     10^3    10^-4.5  <  in <= 10^-1.5
%               ''      1       10^-1.5  <  in <  10^1.5
%       kilo    'k'     10^-3   10^1.5   <= in <  10^4.5
%       mega    'M'     10^-6   10^4.5   <= in <  10^7.5
%       giga    'G'     10^-9   10^7.5   <= in <  10^10.5
%       tera    'T'     10^-12  10^10.5  <= in <  10^13.5
%               ''      1       10^13.5  <= in
%
%	Output parameters:
%		label : prefix label
%       scale : scaling factor
%
%	Input parameters:
%		in : number
%
%	Example:
%       % 10000 Hz can also be displayed as 10 kHz
% 		in = 10000;
%       [label,scale] = fMetricPrefix(in);
%       disp([num2str(in*scale) ' ' label 'Hz'])
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

exponent = round(log10(in)/3);
scale = (1e-3)^exponent;
switch exponent
    case -5,    label='f';
    case -4,    label='p';
    case -3,    label='n';
    case -2,    label='\mu';
    case -1,    label='m';
    case 1,     label='k';
    case 2,     label='M';
    case 3,     label='G';
    case 4,     label='T';        
    otherwise,  label=''; scale=1;
end

%}