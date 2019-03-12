clear all;
run('VLFEATROOT/toolbox/vl_setup');
path = '.\new_data';
folder_list = dir(path);
for i = 3:size(folder_list)
    imglist = dir(fullfile(path,folder_list(i).name));
    for j = 3:size(imglist)
        filename = fullfile(path,folder_list(i).name,imglist(j).name,'test4.png');
        I=imread(filename);
        I = imresize(I,[656,875]);
        I = I(120:519,231:630,:);
        %for ALEXNET
        I = imresize(I,[227,227]);
        name = [fullfile(path,folder_list(i).name),num2str(j-2+25,'%02d'),'.png'];
        imwrite(I,name);
    end
end

imshow(I);


