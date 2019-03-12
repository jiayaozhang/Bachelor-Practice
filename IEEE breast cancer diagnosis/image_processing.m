clear;
path = '.\newdata';
numTrainee = 25;
folder_list = dir(path);
for i = 5:size(folder_list)
    image_list = dir(fullfile(path,folder_list(i).name,'\*.gif'));
    image_num = size(image_list,1);
    
    for j = 1:size(image_list)
            filename = fullfile(path,folder_list(i).name,image_list(j).name);
            
%             Image = imread(filename);
            segmentation(filename,image_list(j).name,i);
            
    end
end