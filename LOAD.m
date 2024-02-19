import cv2.*
import matplotlib.pyplot.*

name = input('Input your file name (.png):', 's');
k = (10 ^ -6) * 9;
component = [];
channels = [];
con_type = 0;
lowpass_coef = 75;
highpass_coef = 21;

unzip(strcat(name, '.zip'), name);

[b_f, g_f, r_f, scale] = load(name);
image1 = rgb2gray(cat(3, b_f, g_f, r_f));

b = fourier_transform(b_f(1:end-1,1:end-1));
g = fourier_transform(g_f(1:end-1,1:end-1));
r = fourier_transform(r_f(1:end-1,1:end-1));

[lowpass_filter, filter_l] = lowpass(lowpass_coef, int32(image1));
b_l = b .* lowpass_filter;
g_l = g .* lowpass_filter;
r_l = r .* lowpass_filter;

[highpass_filter, filter_h] = highpass(highpass_coef, int32(image1));
b_h = b .* highpass_filter;
g_h = g .* highpass_filter;
r_h = r .* highpass_filter;

% MATLAB plotting
figure('Position', [100, 100, 1200, 800]);
subplot(2, 3, 1), imshow(filter_h, []), title('highpass gaussian kernel');
subplot(2, 3, 2), imshow(filter_l, []), title('lowpass gaussian kernel');

% Create a figure with specific size
figure('Position', [100, 100, 1600, 640]);

% Display the original image in the third subplot of a 2x3 grid
subplot(2, 3, 3);
mag = cat(3, show_magnitude(b, 20), show_magnitude(g, 20), show_magnitude(r, 20));
imshow(cv2.cvtColor(mag, cv2.COLOR_BGR2RGB), 'Colormap', gray);
title('original');

% Calculate magnitude for low and high frequencies
mag_l = cat(3, show_magnitude(b_l, 20), show_magnitude(g_l, 20), show_magnitude(r_l, 20));
mag_h = cat(3, show_magnitude(b_h, 20), show_magnitude(g_h, 20), show_magnitude(r_h, 20));

% Create another figure with specific size
figure('Position', [100, 100, 960, 640]);

% Display the highpass image in the first subplot of a 2x3 grid
subplot(2, 3, 1);
imshow(cv2.merge(mag_h), 'Colormap', gray);
title('highpass');

% Display the lowpass image in the second subplot of a 2x3 grid
subplot(2, 3, 2);
imshow(cv2.merge(mag_l), 'Colormap', gray);
title('lowpass');


% Merging channels and converting to uint8
l = uint8(cat(3, inverse_fourier(b_l), inverse_fourier(g_l), inverse_fourier(r_l)));
h = uint8(cat(3, inverse_fourier(b_h), inverse_fourier(g_h), inverse_fourier(r_h)));
without_fourier_image = uint8(cat(3, inverse_fourier(b), inverse_fourier(g), inverse_fourier(r)));

% Creating figures and resizing images
figure('Position', [100, 100, 1200, 600]);
l_final = imresize(l, scale);
h_final = imresize(h, scale);
without_fourier_image_final = imresize(without_fourier_image, scale);

% Applying fourier_uv_sharpening and converting to int
[final, fourier_image, reduced_uv_fourier] = fourier_uv_sharpening(int32(l_final), k);

% Creating another figure
figure('Position', [100, 100, 1200, 800]);
subplot(2, 3, 1), imshow(uint8(fourier_image), []), title('fourier');
subplot(2, 3, 2), imshow(uint8(reduced_uv_fourier), []), title('reduced_uv_fourier');

% Applying threshold and combining images
h_final(h_final < 50) = 0;
result = int32(final) + int32(h_final) * 0.8;
result = uint8(clipping(result));

% Creating another figure and displaying images
figure('Position', [100, 100, 800, 600]);
subplot(2, 3, 1), imshow(imread(strcat(name, '.png'))), title('Original image');
subplot(2, 3, 2), imshow(l_final), title('highpass image');
subplot(2, 3, 3), imshow(h_final), title('lowpass image');
subplot(2, 3, 4), imshow(without_fourier_image_final), title('without_fourier_image_final');
subplot(2, 3, 5), imshow(final), title('final (no highpass)');
subplot(2, 3, 6), imshow(result), title('Results');

function img_normalized = normalize(img)
    img_normalized = (img - min(img(:))) * (255 / (max(img(:)) - min(img(:))));
end

function img_normalized_binary = normalize_binary(img)
    img_normalized_binary = (img - min(img(:))) * (1 / (max(img(:)) - min(img(:))));
end

function kernel = get_gaussian_kernel(l, coff, sig)
    limit = int32(l / coff);
    raw = zeros(l, l);
    for i = -limit:limit
        for j = -limit:limit
            raw(i + limit + 1, j + limit + 1) = exp(-((i ^ 2) + (j ^ 2)) / (2 * (sig ^ 2)));
        end
    end
    kernel = raw;
end

function img_magnitude = show_magnitude(img, n)
    img_magnitude = uint8(n * log(abs(img) + 1));
end

function img_clipped = clipping(img)
    img_clipped = uint8(min(max(img, 0), 255));
end

function [result, normalized_result] = highpass(sigma, image_int)
    [h, w] = size(image_int);
    gauss_kernel = get_gaussian_kernel(max(size(image_int)) + 1, 2, sigma);
    result = normalize_binary(gauss_kernel);
    result = 1 - result;
    dif = int32(abs(h - w) / 2);
    if h > w
        result = result(1:h, dif+1:dif+w);
    elseif w > h
        result = result(dif+1:dif+h, 1:w);
    else
        result = result(1:h-1, 1:w-1);
    end
    normalized_result = normalize(result);
end

function [result, normalized_result] = lowpass(sigma, image_int)
    [h, w] = size(image_int);
    gauss_kernel = get_gaussian_kernel(max(size(image_int)) + 1, 2, sigma);
    result = normalize_binary(gauss_kernel);
    dif = int32(abs(h - w) / 2);
    if h > w
        result = result(1:h, dif+1:dif+w);
    elseif w > h
        result = result(dif+1:dif+h, 1:w);
    else
        result = result(1:h-1, 1:w-1);
    end
    normalized_result = normalize(result);
end

function reduced_uv_fourier_image = uv_main_transform(img)
    
    reduced_uv_fourier_image = zeros(size(img));
    [h, w] = size(img);
    for i = 1:h
        for j = 1:w
            reduced_uv_fourier_image(i, j) = (((i - (h / 2)) ^ 2) + ((j - (w / 2)) ^ 2)) * img(i, j);
        end
    end
    reduced_uv_fourier_image = reduced_uv_fourier_image * (4 * (pi ^ 2));
end

function img_back = inverse_fourier(img)
    f_ishift = ifftshift(img);
    img_back = ifft2(f_ishift);
    img_back = abs(img_back);
    img_back = max(min(img_back, 255), 0);
    img_back = int32(img_back);
end

function fshift = fourier_transform(img)
    f = fft2(img);
    fshift = fftshift(f);
end

function [fourier_uv_final, normalized_inverted_fourier_uv, magnitude_fourier] = fourier_uv_sharpening(image_int, k)
    fourier_image = fft2(image_int);
    reduced_uv_fourier_image = uv_main_transform(fourier_image);
    inverted_fourier_uv = ifft2(reduced_uv_fourier_image);
    fourier_uv_final = im2uint8(imclip(image_int + k * inverted_fourier_uv));
    normalized_inverted_fourier_uv = mat2gray(inverted_fourier_uv);
    magnitude_fourier = log(1 + abs(fftshift(reduced_uv_fourier_image)));
    magnitude_fourier = imresize(magnitude_fourier, 20);
end

function clipped_image = imclip(image)
    clipped_image = min(max(image, 0), 255);
end


function [b, g, r, scale] = load(name)
    config = load([name '/config.npy']);
    ch1 = config(1);
    ch2 = config(2);
    ch3 = config(3);
    con_type = double(config(4));
    scale = double(config(5));
    channels = {ch1, ch2, ch3};
    
    function img = load_channel(color)
        im_u = imread([name '/u_' color '.png']);
        im_v = imread([name '/v_' color '.png']);
        ss = load([name '/s_' color '.npy']);
        max_values = load([name '/max_' color '.npy']);
        mu = max_values(1);
        mv = max_values(2);
        th = max_values(3);
        component(end+1) = th;
        im_u = double(im_u);
        im_v = double(im_v);
        uu = ((im_u - con_type) / con_type) * mu;
        vv = ((im_v - con_type) / con_type) * mv;
        img = uu * diag(ss) * vv;
    end

    b = uint8(clip(round(load_channel(channels{1})), 0, 255));
    g = uint8(clip(round(load_channel(channels{2})), 0, 255));
    r = uint8(clip(round(load_channel(channels{3})), 0, 255));
end

function result = clip(val, minval, maxval)
    result = min(max(val, minval), maxval);
end

