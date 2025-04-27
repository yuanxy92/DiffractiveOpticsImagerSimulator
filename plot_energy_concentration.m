clear;
close all;

% read data
datadir = 'D:/Code/Aurora_papers/results/in_glass_20240723_f_1150um_lambdascan_no_high_fre/10% noise';
outname = 'energy_no_high_noise_0.1';
% lambda_xaxis = 400:10:700;
lambda_xaxis = [462, 511, 606];
energy_curves = zeros(4, 10, length(lambda_xaxis));

dist = 1.15;
n = 1.45;
theta_idx = 0;
for theta = 0:10:30
    theta_idx = theta_idx + 1;
    lambda_idx = 0;
    for lambda = lambda_xaxis
        lambda_idx = lambda_idx + 1;
        matname = sprintf('%s/lambda_%3.1f_dist_%1.3f_theta_%1.1f_phi_0.0_n_%g.mat', datadir, lambda, dist, theta, n);
        energy_data = load(matname);
        for size_idx = 1:10
            energy_curves(theta_idx, size_idx, lambda_idx) = energy_data.energy_percentage{size_idx}.intensity_percentage;
        end
    end
end


legend_labels = string(21:20:201);

f = figure;
f.Position = [50 50 1000 1000];

subplot(2, 2, 1);
plot(lambda_xaxis, squeeze(energy_curves(1, :, :))');
legend(legend_labels);
title('Incident angle: 0 degree')

subplot(2, 2, 2);
plot(lambda_xaxis, squeeze(energy_curves(2, :, :))');
legend(legend_labels);
title('Incident angle: 10 degree')

subplot(2, 2, 3);
plot(lambda_xaxis, squeeze(energy_curves(3, :, :))');
legend(legend_labels);
title('Incident angle: 20 degree')

subplot(2, 2, 4);
plot(lambda_xaxis, squeeze(energy_curves(4, :, :))');
legend(legend_labels);
title('Incident angle: 30 degree')

saveas(gcf, sprintf('%s/%s.pdf', datadir, outname));

