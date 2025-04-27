function img_filtered = deartifact(img)
% img = imread('C:/Projects/Code/PyUVCCamera/data/data_0829_9/camera_6_frame_60.jpg');
    img_filtered = uint8(zeros(size(img)));
    for ch = 1:3
        img_ch = img(:, :, ch);
        img_ch_f0 = fftshift(fft2(img_ch));
        
        base_filter = zeros(size(img_ch_f0));
        base_filter(51:350, 101) = 1;
        base_filter(51:350, 301) = 1;
        gaussian_k = fspecial('gaussian', 9, 2.5);
        
        base_filter_gaussian = conv2(base_filter, gaussian_k, 'same');
        base_filter_gaussian = 1 - base_filter_gaussian / max(base_filter_gaussian(:));
        
        img_ch_f = img_ch_f0 .* base_filter_gaussian;
        img_ch_f = ifftshift(img_ch_f);
        img_ch_filtered = uint8(real(ifft2(img_ch_f)));
        img_filtered(:, :, ch) = img_ch_filtered;
    end
end

% figure;
% subplot(1,4,1);imshow(img_ch);
% subplot(1,4,2);imshow(log(abs(img_ch_f0)), []);
% subplot(1,4,3);imshow(base_filter_gaussian, []);
% subplot(1,4,4);imshow(img_ch_filtered, []);