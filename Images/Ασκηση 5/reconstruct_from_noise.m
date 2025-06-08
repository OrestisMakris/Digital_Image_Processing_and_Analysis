%% Image Restoration and Deconvolution using MATLAB TOOLBOX
% Απαίτηση: Image Processing Toolbox
% Προϋπόθεση: έχετε στη MATLAB διαδρομή αρχείο psf.p, που υλοποιεί τη συνάρτηση psf(X).
% Author:  Orestis Antonis Makris AM-1084156

%% Μέρος Α: Υποβάθμιση με θόρυβο και φίλτρο Wiener

clear all;
close all;

orig = im2double(imread('new_york.png'));

% whuite SNR = 10 dB

snr_db = 10;
noisy = imnoise(orig, 'gaussian', 0, 10 ...
    ^(-snr_db/10));

% estimate snr
noise_var = 10^(-snr_db/10);

% wiener 

PSF_delta = zeros(3,3);
PSF_delta(2,2) = 1;
K = noise_var / var(orig(:));
wiener_known = deconvwnr(noisy, PSF_delta, K) ;

% regularized inverse no prior nowledf of k
wiener_unknown = deconvreg(noisy, PSF_delta) ;

fig1 = figure('Units','normalized','OuterPosition',[0 0 1 1]);
t1 = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile; imshow(orig); title('Αρχική','FontSize',14);
nexttile; imshow(noisy); title('Noise (SNR=10dB)','FontSize',14);
nexttile; imshow(wiener_known); title('Wiener (γνωστό K)','FontSize',14);
nexttile; imshow(wiener_unknown); title('Regularized inverse (άγνωστο)','FontSize',14);

