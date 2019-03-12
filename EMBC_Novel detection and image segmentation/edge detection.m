I = imread('cell.tif');%��ȡͼ��
figure;  imshow(I), %��ʾͼ��
title('original image');
text(size(I,2),size(I,1)+15, ...
    'Image courtesy of Alan Partin', ...
    'FontSize',7,'HorizontalAlignment','right');
text(size(I,2),size(I,1)+25, ....
    'Johns Hopkins University', ...
    'FontSize',7,'HorizontalAlignment','right');
[junk threshold] = edge(I, 'sobel');%��Ե���
fudgeFactor = .5;
BWs = edge(I,'sobel', threshold * fudgeFactor);%�ı�����ټ���Ե
figure; subplot(221),
imshow(BWs), %��ʾ��ֵͼ��
title('binary gradient mask');
se90 = strel('line', 3, 90);%��ֱ�����ԽṹԪ��
se0 = strel('line', 3, 0);%ˮƽ�����ԽṹԪ��
BWsdil = imdilate(BWs, [se90 se0]);%��ͼ���������
subplot(222); imshow(BWsdil), %��ʾ���ͺ�Ķ�ֵͼ��
title('dilated gradient mask');
BWdfill = imfill(BWsdil, 'holes');%��ͼ��������
subplot(223); imshow(BWdfill);%��ʾ����Ķ�ֵͼ��
title('binary image with filled holes');
BWnobord = imclearborder(BWdfill, 4);%ȥ���߽��ϵ�ϸ��
subplot(224); imshow(BWnobord), %��ʾȥ���߽�ϸ����Ķ�ֵͼ��
title('cleared border image');
seD = strel('diamond',1);%���νṹԪ��
BWfinal = imerode(BWnobord,seD);%��ʴͼ��
BWfinal = imerode(BWfinal,seD);%��ʴͼ��
figure; subplot(121)
imshow(BWfinal), %��ʾ�ָ���ͼ��
title('segmented image');
BWoutline = bwperim(BWfinal);%ȡ��ϸ���ı߽�
Segout = I;
Segout(BWoutline) = 255;%ϸ���߽紦��255
subplot(122), imshow(Segout), %��ԭʼͼ������ʾ�߽�
title('outlined original image');
