%% Μέρος Β: Απόκριση PSF & Αντίστροφο Φιλτράρισμα (MATLAB TOOLBOX)


orig = im2double(imread('new_york.png'));

%  Εκτίμηση κρουστικής απόκρισis
[m,n] = size(orig);
delta = zeros(m,n);
delta(ceil(m/2), ceil(n/2)) = 1;
h = psf(delta);  % Υπολογισμός h μέσω psf P-code

% Απόκριση συχνότητας της PSF
H = fftshift(fft2(h));
figure('Name','PSF Frequency Response');
imshow((abs(H)), []);
colorbar;
title('PSF Frequency Response (log magnitude)','FontSize',14);

% Αντίστροφο φίλτρο στο πεδίο συχνοτήτων με κατώφλι και MSE
thresholds = logspace(-4, -1, 10);
mse_vals = zeros(size(thresholds));
F_orig_blur = fft2(psf(orig));  % Συγχώνευση θόλωμα+FFT
for i = 1:numel(thresholds)
    thr = thresholds(i);
    H_inv = zeros(size(H));
    mask = abs(H) >= thr;
    H_inv(mask) = 1 ./ H(mask);
    recon = real(ifft2(ifftshift(H_inv) .* F_orig_blur));
    mse_vals(i) = immse(recon, orig);
end

% Σχεδίαση MSE &threshold
figure('Name','MSE Vs Threshold');
semilogx(thresholds, mse_vals, 'o-');
grid on;
xlabel('Threshold');
ylabel('MSE','FontSize',12);
title('MSE Vs Threshold for Inverse Filtering') ;

