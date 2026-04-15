clear all
close all
clc

% --- User Configuration ---
fprintf('Loading measurement image and sensing matrix (1024×1024×41)...\n');
% Load hyperspectral mask
load('mask_400_800-1024_1024_41_jianhuanjing_chubeijing_qe_integral.mat');
mask_400_800 = single(mask_400_800);
mask_400_800 = max(mask_400_800, 0);
mask_400_800 = mask_400_800 / max(max(max(mask_400_800)));

% --- User Configuration: Auto-detect FIT/PNG file and read ---
imageFilePath = "falling leaves16.png"; % Only modify the filename here; supports .fit/.fits/.png

% Get file extension and auto-detect format
[~, ~, fileExt] = fileparts(imageFilePath);
fileExt = lower(fileExt); 

if strcmp(fileExt, '.fit') || strcmp(fileExt, '.fits')
    fprintf('Detected FIT/FITS file, reading with fitsread...\n');
    measurement = fitsread(imageFilePath);
elseif strcmp(fileExt, '.png')
    fprintf('Detected PNG file, reading with imread...\n');
    measurement = imread(imageFilePath);
else
    error('Unsupported file format! Only .fit / .fits / .png are supported.');
end

measurement = double(measurement);

[imgH, imgW, ~] = size(measurement);

if imgH == 1080 && imgW == 1920
    fprintf('Detected image size: 1080×1920 → Cropping measurement to 1024×1024...\n');
    H = 1024;
    W = 1024;
    % 1080×1920：
    measurement = measurement(10:1033, 801:1824, :);

elseif imgH == 640 && imgW == 640
    fprintf('Detected image size: 640×640 → Cropping mask to 640×640...\n');
    H = 640;
    W = 640;
    % 640×640：
    mask_400_800 = mask_400_800(2:641, 1:640, :);
else
    error('Only 1080×1920 or 640×640 images are supported!');
end
% ======================================================================

measurement = measurement / max(measurement(:));

% --- 1. Import ONNX Model ---
% Ensure the ONNX file is in the current MATLAB path
onnxFilePath = 'zhe_400_800_41_8_8_channel32_4e-4_zuheloss_shiwai_IR_integral_rate5_psnr43.22.onnx';
net = importNetworkFromONNX(onnxFilePath);

% --- 2. Prepare Input Data ---
C_input = 1;      % Number of channels for the first input
C_mask = 41;      % Number of channels for the second input (spectral channels)
N = 1;            % Batch size

% Create random dummy input data
% Note: MATLAB default dimension order is H x W x C x N
dummy_measurement_mat = rand(H, W, C_input, N, 'single');
dummy_mask_mat = rand(H, W, C_mask, N, 'single');

% Convert to dlarray and specify dimension labels 'BCSS'
dummy_measurement_dl = dlarray(permute(dummy_measurement_mat, [4, 3, 1, 2]), 'BCSS');
dummy_mask_dl = dlarray(permute(dummy_mask_mat, [4, 3, 1, 2]), 'BCSS');

measurement_dl = dlarray(permute(single(measurement), [4, 3, 1, 2]), 'BCSS');
mask_dl = dlarray(permute(single(mask_400_800), [4, 3, 1, 2]), 'BCSS');

% --- 3. Run on GPU (if available) ---
if canUseGPU
    fprintf('Initializing network model...\n');
    dummy_measurement_dl = gpuArray(dummy_measurement_dl);
    dummy_mask_dl = gpuArray(dummy_mask_dl);
    measurement_dl = gpuArray(measurement_dl);
    mask_dl = gpuArray(mask_dl);
end

% --- Initialize dlnetwork ---
net = initialize(net, dummy_measurement_dl, dummy_mask_dl);

% --- 4. Run Prediction ---
fprintf('Running prediction...\n');
reconstructed_hsi_dl = predict(net, measurement_dl, mask_dl);

% --- 5. Process Output ---
reconstructed_hsi = gather(extractdata(reconstructed_hsi_dl));

fprintf('Prediction completed!\n');

% --- 6. Visualization ---
fprintf('Visualizing results...\n');
figure;
% Main title
sgtitle('Measurement vs Reconstructed HSI', 'FontSize', 18, 'FontWeight', 'bold');
subplot(2, 3, 1);
imagesc(measurement);
colorbar;
colormap("gray")
title('Measurement', 'FontSize', 14, 'FontWeight', 'bold');
axis image;
axis off;

rgb = cat(3, reconstructed_hsi(:, :, 26), reconstructed_hsi(:, :, 14), reconstructed_hsi(:, :, 7));
subplot(2, 3, 2);
imagesc(rgb);
colorbar;
title('Synthesized RGB', 'FontSize', 14, 'FontWeight', 'bold');
axis image;
axis off;

for i = 1:4
    subplot(2, 3, i + 2);
    imagesc(reconstructed_hsi(:, :, (i - 1) * 10 + 1));
    colorbar;
    colormap("gray")
    title([num2str(400 + (i - 1) * 100), ' nm'], 'FontSize', 14, 'FontWeight', 'bold');
    axis image;
    axis off;
end

figure
imagesc(rgb);
rgbv2 = 2 * rgb;
imshow(rgbv2);
title('Synthesized RGB', 'FontSize', 14, 'FontWeight', 'bold');
axis image;
axis on;