name = input('Input your file name (.png):', 's');

channels = {'b','g','r'};

comp = 450;

scale = 1.4;
con_type = (2^15)-1;
img = imread(strcat(name, '.png'));
img = imresize(img, 1/scale);

% Create directory if it does not exist
if ~exist(name, 'dir')
    mkdir(name);
else
    fprintf('Directory already exists.\n');
end


save(name, img,channels,con_type,scale);
zip(strcat(name, '.zip'), name);
rmdir(name, 's');

function save(name, img,channels,con_type,scale)
    config = channels;
    config(end+1) = con_type;
    config(end+1) = scale;
    save(strcat(name, '/config.mat'), 'config');
    function save_channel(th, color, u, s, v)
        uu = u(:, 1:th);
        vv = v(1:th, :);
        ss = s(1:th);
        mu = max(abs(uu), [], 'all');
        mv = max(abs(vv), [], 'all');
        im_u = uint16(round(((uu / mu) * con_type) + con_type));
        im_v = uint16(round(((vv / mv) * con_type) + con_type));
        save(strcat(name, '/max_', color, '.mat'), 'mu', 'mv', 'th');
        save(strcat(name, '/s_', color, '.mat'), 'ss');
        imwrite(im_u, strcat(name, '/u_', color, '.png'));
        imwrite(im_v, strcat(name, '/v_', color, '.png'));
    end
    for index = 1:length(channels)
        ch = channels(index);
        [u, s, v] = svd(img(:, :, index), 'econ');
        threshhold = length(s);
        for ind = 1:length(s)
            if s(ind) < comp
                threshhold = ind - 1;
                break;
            end
        end
        save_channel(threshhold, ch, u, s, v);
    end
end


