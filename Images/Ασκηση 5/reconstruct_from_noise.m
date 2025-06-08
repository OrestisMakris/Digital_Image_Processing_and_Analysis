%% Image Restoration and Deconvolution using MATLAB TOOLBOX
% Απαίτηση: Image Processing Toolbox
% Προϋπόθεση: έχετε στη MATLAB διαδρομή αρχείο psf.p, που υλοποιεί τη συνάρτηση psf(X).
% Author:  Orestis Antonis Makris AM-1084156

%% Μέρος Α: Υποβάθμιση με θόρυβο και φίλτρο Wiener


orig = im2double(imread('new_york.png'));

% whuite SNR = 10 dB

snr_db = 30;
noisy = imnoise(orig, 'gaussian', 0, 10^(-snr_db/10));

% estimate snr
noise_var = 10^(-snr_db/10);

% wiener 

PSF_delta = zeros(3,3);
PSF_delta(2,2) = 1;
K = noise_var / var(orig(:));
wiener_known = deconvwnr(noisy, PSF_delta, K) ;

% regularized inverse no prior nowledf of k
wiener_unknown = deconvreg(noisy, PSF_delta);

fig1 = figure('Units','normalized','OuterPosition',[0 0 1 1]);
t1 = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile; imshow(orig); title('Αρχική','FontSize',14);
nexttile; imshow(noisy); title('Νορυθμισμένη (SNR=10dB)');
nexttile; imshow(wiener_known); title('Wiener (γνωστό K)');
nexttile; imshow(wiener_unknown); title('Regularized inverse (άγνωστο)');

%% Μέρος Β: Απόκριση PSF & Αντίστροφο Φιλτράρισμα

[m,n] = size(orig);
delta = zeros(m,n);
delta(ceil(m/2), ceil(n/2)) = 1;
h = psf(delta);
H = fftshift(fft2(h));
fig2 = figure('Units','normalized','OuterPosition',[0 0 1 1]);
imshow(log1p(abs(H)), []); colorbar; title('PSF Frequency Response (log magnitude)','FontSize',14);

blurred = psf(orig);
fig3 = figure('Units','normalized','OuterPosition',[0 0 1 1]);
t2 = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; imshow(orig); title('Original','FontSize',14);
nexttile; imshow(blurred); title('Blurred by PSF','FontSize',14);

thresholds = logspace(-4, -1, 10);
mse_vals = zeros(size(thresholds));
F_blur = fft2(blurred);
for i = 1:length(thresholds)
    thr = thresholds(i);
    H_inv = zeros(size(H));
    mask = abs(H) >= thr;
    H_inv(mask) = 1 ./ H(mask);
    recon = real(ifft2(ifftshift(H_inv) .* F_blur));
    mse_vals(i) = immse(recon, orig);
end

fig4 = figure('Units','normalized','OuterPosition',[0 0 1 1]);
semilogx(thresholds, mse_vals, 'o-'); grid on;
xlabel('Threshold','FontSize',14); ylabel('MSE','FontSize',14);
title('MSE vs Threshold','FontSize',14);

H_inv_full = 1 ./ H;
recon_no_thr = real(ifft2(ifftshift(H_inv_full) .* F_blur));
fig5 = figure('Units','normalized','OuterPosition',[0 0 1 1]);
t3 = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; imshow(orig); title('Original','FontSize',14);
nexttile; imshow(recon_no_thr); title('Inverse Filter (no threshold)','FontSize',14);
