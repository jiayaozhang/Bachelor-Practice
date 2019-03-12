function [ I ] = segmentation( path, name , fold)
% clear;
fold
    
%[I,map] = imread('.\data\C_0150_1.RIGHT_MLO.LJPEG.1_highpass.gif');
% [I,map] = imread('.\data\C_0001_1.RIGHT_CC.LJPEG.1_highpass.gif');
% [I,map] = imread('.\data\C_0152_1.RIGHT_MLO.LJPEG.1_highpass.gif');
[I, map] = imread(path);
% I = imresize(I,[353,225]);
if ~isempty( map )     
   I = ind2rgb( I, map );         %将索引图像数据转为RGB图像数据
end 
% I_origin = I;
I = rgb2gray(I);
% if contains(name,'LEFT') && fold<5
%     I = flip(I,2);
% end
% if contains(name,'A') || contains(name,'D') && contains(name,'RIGHT') && fold == 5
%     I = flip(I,2);
% end
if contains(name,'RIGHT')
    I = flip(I,2);
end
if contains(name,'4573') || contains(name,'4586') ||contains(name,'4602') ||contains(name,'4603') ||contains(name,'4604') || contains(name,'4605') || contains(name,'4606') ||contains(name,'4609')
    I = flip(I,2);
end
[height,weight] = size(I);  
for h = 2:height-1
    flag = 0;
    for w = weight-1:-1:8
        temp1 = I(h,w)+I(h,w-1)+I(h,w-2)+I(h,w-3)+I(h,w-4)+I(h,w-5)+I(h,w-6);
        temp2 = I(h+1,w)+I(h+1,w-1)+I(h+1,w-2)+I(h+1,w-3)+I(h+1,w-4)+I(h+1,w-5)+I(h+1,w-6);
        temp3 = I(h-1,w)+I(h-1,w-1)+I(h-1,w-2)+I(h-1,w-3)+I(h-1,w-4)+I(h-1,w-5)+I(h-1,w-6);
        temp = temp1+temp2+temp3;
        if temp<0.50
            flag = 1;
        end
        if I(h,w)== 0 || I(h,w)== 0.298936021293776
%             I(h,w) = (I(h,w+1)*0.6+I(h,w+2)*0.3+I(h+1,w)*0.05+I(h+1,w+1)*0.05);
            I(h,w) = I(h,w+1);
        end
        if flag == 1
            I(h,w) = 0;
        end
    end
end
I =  I(2:height,9:weight-5);
pad = zeros(500,500);

[height,weight] = size(I);
% figure;
% imshow(pad);
pad(250-height/2:250+height/2-1,250-weight/2:250+weight/2-1) = I(:,:);
I = pad;


% figure;
imshow(I);
name = fullfile('out',num2str(fold),['out',name]);
% strrep(name,'gif','jpg');
% name = name(1:-3)
name = [name,'.jpg']

imwrite(I,name);

end

