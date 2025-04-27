clear;
close all;
fclose all;

for index = 3:8
    input_folder = sprintf('C:/Projects/Code/PyUVCCamera/data/data_0829_%d', index); 
    output_folder = sprintf('C:/Projects/Code/Aurora_papers/data/arm_6cam/data_0829_%d_undis', index); 
    mkdir(output_folder)
    file_list = dir(fullfile(input_folder, '*.jpg')); 
    load('camera_param/camera_params.mat');
    
    for i = 1:length(file_list)
        filename = fullfile(input_folder, file_list(i).name);
        img = imread(filename);
        img = deartifact(img);
        corrected_img = undistortImage(img, cameraParams); 
        corrected_img = imrotate(corrected_img, 90);
        
        [~, name, ext] = fileparts(file_list(i).name);
        ext = '.png';
        output_filename = fullfile(output_folder, [name, '_corrected', '.png']);
        
        imwrite(corrected_img, output_filename); 
        % fprintf('Processed: %s\n', file_list(i).name); 
    end
    
    fprintf('%s processing complete.\n', input_folder);

end