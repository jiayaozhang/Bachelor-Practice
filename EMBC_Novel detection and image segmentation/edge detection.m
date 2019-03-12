I = imread('cell.tif');%读取图像
figure;  imshow(I), %显示图像
title('original image');
text(size(I,2),size(I,1)+15, ...
    'Image courtesy of Alan Partin', ...
    'FontSize',7,'HorizontalAlignment','right');
text(size(I,2),size(I,1)+25, ....
    'Johns Hopkins University', ...
    'FontSize',7,'HorizontalAlignment','right');
[junk threshold] = edge(I, 'sobel');%边缘检测
fudgeFactor = .5;
BWs = edge(I,'sobel', threshold * fudgeFactor);%改变参数再检测边缘
figure; subplot(221),
imshow(BWs), %显示二值图像
title('binary gradient mask');
se90 = strel('line', 3, 90);%垂直的线性结构元素
se0 = strel('line', 3, 0);%水平的线性结构元素
BWsdil = imdilate(BWs, [se90 se0]);%对图像进行膨胀
subplot(222); imshow(BWsdil), %显示膨胀后的二值图像
title('dilated gradient mask');
BWdfill = imfill(BWsdil, 'holes');%对图像进行填充
subplot(223); imshow(BWdfill);%显示填充后的二值图像
title('binary image with filled holes');
BWnobord = imclearborder(BWdfill, 4);%去除边界上的细胞
subplot(224); imshow(BWnobord), %显示去除边界细胞后的二值图像
title('cleared border image');
seD = strel('diamond',1);%菱形结构元素
BWfinal = imerode(BWnobord,seD);%腐蚀图像
BWfinal = imerode(BWfinal,seD);%腐蚀图像
figure; subplot(121)
imshow(BWfinal), %显示分割后的图像
title('segmented image');
BWoutline = bwperim(BWfinal);%取得细胞的边界
Segout = I;
Segout(BWoutline) = 255;%细胞边界处置255
subplot(122), imshow(Segout), %在原始图像上显示边界
title('outlined original image');
