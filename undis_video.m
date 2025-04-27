clear;
close all;
fclose all;

input_videoname = 'C:/Projects/dataset/OV6946_20240827/LSK_inside.mp4'; 
output_videoname = 'C:/Projects/dataset/OV6946_20240827/LSK_inside_undis.mp4'; 

load('camera_param/camera_params.mat'); % 这里假设相机参数保存在camera_params.mat文件中

videoReader = VideoReader(input_videoname);
videoWriter = VideoWriter(output_videoname, 'MPEG-4');
open(videoWriter);

while hasFrame(videoReader)
    % Read the current frame
    img = readFrame(videoReader);
    % 畸变矫正
    corrected_img = undistortImage(img, cameraParams); % 使用undistortImage函数进行畸变矫正
    % 保存矫正后的图片
    writeVideo(videoWriter, corrected_img);
end

close(videoWriter);
disp('Batch processing complete.'); % 显示处理完成
