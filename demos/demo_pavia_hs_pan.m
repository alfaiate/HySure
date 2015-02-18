% This is a demo file that exemplifies the use of the HySure algorithm.
% See the file README for more information.
% 
% It corresponds to the example given in [1] using dataset B (Pavia
% University dataset), with the fusion of a simulated hyperspectral image 
% with a simulated panchromatic image. 
% 
% See [1] for more details, but essentially we used the original image 
% (with high resolution both in the spatial and in the spectral) as ground 
% truth. To create a hyperspectral image, we spatially blurred the ground  
% truth one, and then downsampled the result by a factor of 4 in each 
% direction. We then filtered it with the Starck-Murtagh filter. To create 
% the panchromatic/multispectral images, the spectral response of the  
% IKONOS satellite was used. Gaussian noise was added to the hyperspectral 
% image (SNR=30 dB) and to the panchromatic/multispectral images
% (SNR=40 dB).
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
% I. It starts by generating the observed hyperspectral and
% multispectral/panchromatic images. The following parameters can be
% modified to change the data generation:
% 
downsamp_factor = 4; % Downsampling factor
SNRh = 30; % SNR (in dB) for the hyperspectral image
SNRm = 40; % SNR (in dB) for the multispectral/panchromatic image
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
lambda_phi = 1e-2;
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

ms_bands = 1; % Only panchromatic: 1 - pan; 2 - blue; 3 - green; 4 - red;
%                                  5 - NIR

% Load ROSIS image (without the first ten bands and without the last
% collumn)
load ../data/original_rosis
[nl, nc, L] = size(X);
% In matrix form
Z = im2mat(X);
Z = Z/max(Z(:));
clear X;

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
Yh = ConvC(Z, FB, nl);
Yhim_up = mat2im(Yh, nl);
% Add noise
sigmah = sqrt(sum(Yhim_up(:).^2)/(10^(SNRh/10))/numel(Yhim_up));
Yhim_up = Yhim_up + sigmah*randn(size(Yhim_up));
% % % % % % % % % % % Downsampling (version with reduced size)
% Downsampling
Yhim = downsamp_HS(Yhim_up, downsamp_factor, 1);
    
% % % % % % % % % % % % % 
% Simulate the MS/PAN data
% Use IKONOS's spectral response 
% (wavelenghths, pan, blue, green, red, NIR, in nanometers)
load ../data/ikonos_spec_resp.mat
% Map IKONOS' wavelengths into ROSIS (430 - 860 nm) bands
% Find valid interval ikonos \subset rosis
[~, valid_ik_bands] = intersect(ikonos_sp(:,1), 430:860);
no_wa = length(valid_ik_bands);
% Spline interpolation
xx  = linspace(1, no_wa, L);
x = 1:no_wa;
R = zeros(5, L);
for i = 1:5 % 1 - pan; 2 - blue; 3 - green; 4 - red; 5 - NIR
    R(i,:) = spline(x, ikonos_sp(valid_ik_bands,i+1), xx);
end
% Use just the predefined bands
R = R(ms_bands,:);
% Spectral degradation
Ym = R*Z;
% Normalize all channels to 1 (NOTE: this changes the SNRm)
c = zeros(length(ms_bands));
for i=1:length(ms_bands);
    c(i) = max(Ym(i,:));
    Ym(i,:) = Ym(i,:)/c(i);
    R(i,:) =  R(i,:)/c(i);
end
% Add noise
sigmam = sqrt(sum(Ym(:).^2)/(10^(SNRm/10))/numel(Ym));
Ym = Ym + sigmam*randn(size(Ym));
Ymim = mat2im(Ym, nl);

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
intersection = cell(1,length(ms_bands));
intersection{1} = 1:L;
contiguous = intersection;

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
