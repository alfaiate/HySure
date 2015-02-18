% This is a demo file that exemplifies the use of the HySure algorithm.
% See the file README for more information.
% 
% It corresponds to the example given in [1] using dataset C (Paris
% dataset), with the fusion of a simulated hyperspectral image 
% with a simulated panchromatic image. 
% 
% This is a real dataset, with images taken by two instruments on board 
% the EO-1 satellite, Hyperion (hyperspectral) and ALI (pan+multispectral). 
% We needed the hyperspectral image to have lower resolution than the 
% multispectral one, and therefore we first reduced the spatial resolution 
% of the hyperspectral image by blurring it with the Starck-Murtagh filter 
% and downsampling, as described in [1] and using a downsampling factor
% of 3. The original hyperspectral image, before blurring and downsampling,
% was used as ground truth.
% 
% The downsampling factor and SNR are values that can be modified 
% (see below). 
% 
%   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, 
%        “A convex formulation for hyperspectral image superresolution via 
%        subspace-based regularization,” IEEE Trans. Geosci. Remote Sens.,
%        to be publised.

% % % % % % % % % % % % % 
% 
% Version: 1
% 
% Can be obtained online from: https://github.com/alfaiate/HySure
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2015 Miguel Simoes, Jose Bioucas-Dias, Luis B. Almeida 
% and Jocelyn Chanussot
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, version 3 of the License.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% 
% % % % % % % % % % % % % 
clear; close all;
addpath('../src', '../src/utils');
% % % % % % % % % % % % % 
%
% This script has four steps. 
% I. It starts by generating the observed hyperspectral images. 
% The following parameters can be
% modified to change the data generation:
% 
downsamp_factor = 3; % Downsampling factor
SNRh = 30; % SNR (in dB) for the hyperspectral image
% 
% II. Next, it estimates the spectral and spatial response of the sensors.
% The regularization parameters can be adjusted here:
lambda_R = 1e1;
lambda_B = 1e1;
% For the denoising with SVD, we need to specify the number of bands we
% want to keep
p = 10; % Corresponds to variable L_s in [1]; number of endmembers in VCA /
% number of non-truncated singular vectors
%
% III. The data fusion algorithm is then called using the estimated responses
% and the observed data. The following parameters can be
% modified:
% 
basis_type = 'VCA';
lambda_phi = 5e-4;
lambda_m = 1e0;
% 
% IV. In the end, three quality indices are computed using the ground
% truth image. These indices are ERGAS, SAM, and UIQI.
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% I. Observed data (simulation)                                         %
% -----------------------------                                         %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

ms_bands = 1:9; % 9 MS bands

% Load image (without the first ten bands and without the last
% collumn)
load ../data/paris_data_hs_ms
[nl, nc, ~] = size(MSim);
L = size(HSim, 3);

MSp = zeros(size(MSim));
for i=1:length(ms_bands)
    x = MSim(:,:,i);
    xmax  = quantile(x(:),0.999);
    % Normalize to 1
    x = x/xmax;
    MSp(:,:,i) = x;
end
Ymim = MSp;

HSp = zeros(size(HSim));
for i=1:L
    x = HSim(:,:,i);
    xmax  = quantile(x(:),0.999);
    %             xmax  = max(x(:));
    %normalize to 1
    x = x/xmax;
    HSp(:,:,i) = x;
end
Zim = HSp;
        
% % % % % % % % % % % % % 
% Blur kernel
middlel = round((nl+1)/2);
middlec = round((nc+1)/2);
% Blur matrix
B = zeros(nl,nc);
% Starck-Murtagh filter
B(middlel-2:middlel+2, middlec-2:middlec+2) = [1 4 6 4 1; 4 16 24 16 4; 6 24 36 24 6; 4 16 24 16 4; 1 4 6 4 1];
% Circularly center B
B = ifftshift(B);
% Normalize
B = B/sum(sum(B));
% Fourier transform of the filters
FB = fft2(B);

% % % % % % % % % % % % % 
% Simulate the HS data
% Spatial degradation (blur)
Z = im2mat(Zim);
Yh = ConvC(Z, FB, nl);
Yhim_up = mat2im(Yh, nl);
% Add noise
sigmah = sqrt(sum(Yhim_up(:).^2)/(10^(SNRh/10))/numel(Yhim_up));
Yhim_up = Yhim_up + sigmah*randn(size(Yhim_up));
% % % % % % % % % % % Downsampling (version with reduced size)
% Downsampling
Yhim = downsamp_HS(Yhim_up, downsamp_factor, 1);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% II. Spectral and spatial responses estimation                         %
% ---------------------------------------------                         %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% Define the spectral coverage of both sensors, i.e., which HS band
% corresponds to each MS band
% e.g.: MS band 1 - HS bands 1,2,3,4,5
%       MS band 2 - HS bands 6,7,8,9
%       MS band 3 - ...
% Now imagine that there are some bands that are very noisy. We remove them
% from the hyperspectral data cube and build a vector with the number of 
% the bands there were not removed (we call this vector 'non_del_bands').
% For example, if we removed bands 3 and 4, non_del_bands = [1,2,5,6,...]
% We now define a cellarray, called 'intersection',  with length(ms_bands) 
% cells. Each cell corresponds to a multispectral band and will have a 
% vector with the number of the hyperspectral bands that are covered by it.
% Since we removed some bands as well, we need to keep track of the bands
% that are contiguous in the data cube but are not in the sensor.
% We call this other cellarray 'contiguous'. If there are no
% removed bands, it can be set to be the same as 'intersection'.

% Hyperions's spectral coverage can be found here: 
% http://eo1.usgs.gov/sensors/hyperioncoverage
intersection = cell(1,length(ms_bands));
contiguous = cell(1,length(ms_bands));
% We use a vector with the number of the non-deleted bands
% ('non_del_bands').
[~, intersection{1}, contiguous{1}] = intersect(non_del_bands, 9:10);
[~, intersection{2}, contiguous{2}] = intersect(non_del_bands, 11:16);
[~, intersection{3}, contiguous{3}] = intersect(non_del_bands, 18:25);
[~, intersection{4}, contiguous{4}] = intersect(non_del_bands, 28:33);
[~, intersection{5}, contiguous{5}] = intersect(non_del_bands, 42:45);
[~, intersection{6}, contiguous{6}] = intersect(non_del_bands, 49:53);
[~, intersection{7}, contiguous{7}] = intersect(non_del_bands, 106:115);
[~, intersection{8}, contiguous{8}] = intersect(non_del_bands, 141:160);
[~, intersection{9}, contiguous{9}] = intersect(non_del_bands, 193:219);

% Blur's support: [hsize_h hsize_w]
hsize_h = 10;
hsize_w = 10;
shift = 1; % the 'phase' parameter in MATLAB's 'upsample' function
blur_center = 0; % to center the blur kernel according to the simluated data
[V, R_est, B_est] = sen_resp_est(Yhim, Ymim, downsamp_factor, intersection, contiguous, p, lambda_R, lambda_B, hsize_h, hsize_w, shift, blur_center);

% Denoises the original image, since it is quite noisy as well
Z = (V*V')*Z;
% In image form
Zim = mat2im(Z, nl);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% III. Data fusion                                                      %
% ----------------                                                      %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

Zimhat = data_fusion(Yhim, Ymim, downsamp_factor, R_est, B_est, p, basis_type, lambda_phi, lambda_m);

% Denoise the data again with V
Zhat = im2mat(Zimhat);
Zhat_denoised = (V*V')*Zhat;
% In image form
Zimhat = mat2im(Zhat_denoised, nl);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                       %
% IV. Quality indices                                                   %
% -------------------                                                   %
%                                                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

[rmse_total, ergas, sam, uiqi] = quality_assessment(Zim, Zimhat, 0, 1/downsamp_factor);

fprintf('Quality indices:\n RMSE = %2.3f\n ERGAS = %2.3f\n SAM = %2.3f\n UIQI = %2.3f\n', ...
    rmse_total,ergas,sam,uiqi)
