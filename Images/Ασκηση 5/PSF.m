%% Μέρος Β: Απόκριση PSF & Αντίστροφο Φιλτράρισμα (MATLAB TOOLBOX)
% Προϋπόθεση: psf.p (P-code) στον τρέχοντα φάκελο ή στο MATLAB path

orig = im2double(imread('new_york.png'));

% 1. Εκτίμηση κρουστικής απόκρισης (impulse response)
[m,n] = size(orig);
delta = zeros(m,n);
delta(ceil(m/2), ceil(n/2)) = 1;
h = psf(delta);

% 2. Υπολογισμός απόκρισης συχνότητας
H = fftshift(fft2(h));
figure('Units','normalized','OuterPosition',[0 0 1 1]);
imshow(log1p(abs(H)), []);
colorbar;
title('PSF Frequency Response (log magnitude)','FontSize',14);

% 3. Θόλωμα μέσω psf
blurred = psf(orig);
figure('Units','normalized','OuterPosition',[0 0 1 1]);
t = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; imshow(orig); title('Original','FontSize',14);
nexttile; imshow(blurred); title('Blurred by PSF','FontSize',14);

% 4. Αντίστροφο φίλτρο στο πεδίο συχνοτήτων με threshold
defaultThresholds = logspace(-4, -1, 10);
mse_vals = zeros(size(defaultThresholds));
F_blur = fft2(blurred);
for i = 1:length(defaultThresholds)
    thr = defaultThresholds(i);
    H_inv = zeros(size(H));
    mask = abs(H) >= thr;
    H_inv(mask) = 1 ./ H(mask);
    recon = real(ifft2(ifftshift(H_inv) .* F_blur));
    mse_vals(i) = immse(recon, orig);
end
figure('Units','normalized','OuterPosition',[0 0 1 1]);
semilogx(defaultThresholds, mse_vals, 'o-'); grid on;
xlabel('Threshold','FontSize',14);
ylabel('MSE','FontSize',14);
title('MSE vs Threshold','FontSize',14);

% 5. Αντίστροφο φίλτρο χωρίς threshold
H_inv_full = 1 ./ H;
recon_no_thr = real(ifft2(ifftshift(H_inv_full) .* F_blur));
figure('Units','normalized','OuterPosition',[0 0 1 1]);
t2 = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; imshow(orig); title('Original','FontSize',14);
nexttile; imshow(recon_no_thr); title('Inverse Filter (no threshold)','FontSize',14);

