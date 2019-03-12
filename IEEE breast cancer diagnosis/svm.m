clear all;
run('VLFEATROOT/toolbox/vl_setup');
path = '.\photoacoustic_data_crop';
folder_list = dir(path);
% only 3 class and they are alike with each other
% so the cluster should be small
numClusters = 256;
total=0;
%get feature data and label
train_sample = 75;
test_sample = 14;
% data = [];
for i = 3:size(folder_list)
    imglist = dir(fullfile(path,folder_list(i).name));
%     for j = 3:size(imglist)
    for j = 3:test_sample+train_sample+2
        j
        filename = fullfile(path,folder_list(i).name,imglist(j).name);
        I=imread(filename);
        [~,~,RGB]=size(I);
        if RGB==3
            I = rgb2gray(I);
        end
        points = detectSURFFeatures(I);
        point_size=points.size;
        if point_size(1)~=0
            total=total+1;
            new=points;
            point_size=new.size;
            [features, valid_points] = extractFeatures(I,new);
            data(:,total:total+point_size(1)-1)=features';
            label(:,total:total+point_size(1)-1)=repmat([i-2;j-2],1,point_size(1));
            total=total+point_size(1)-1;
        end
    end
end
% using K-means to get the feature words.
[centers, assignments] = vl_kmeans(data, numClusters, 'Initialization', 'plusplus', 'Algorithm', 'Elkan');


test_total = 0;
train_total = 0;
%Train data;
for i = 3:size(folder_list)
    imglist = dir(fullfile(path,folder_list(i).name));
    for j = 3:test_sample+train_sample+2
        j
        filename = fullfile(path,folder_list(i).name,imglist(j).name);
        I=imread(filename);
        [~,~,RGB]=size(I);
        if RGB==3
            I = rgb2gray(I);
        end
        points = detectSURFFeatures(I);
        point_size=points.size;
        if point_size(1)~=0
            [features, valid_points] = extractFeatures(I,points);
            features=features';
            temp=zeros(1,numClusters);
            for k=1:valid_points.Count
                test=features(:,k);
                difference=sum(abs(centers-repmat(test,1,numClusters)));
                [~,position]=min(difference);
                temp(1,position)=temp(1,position)+1;
                difference(position)=9999;
                
                [~,position]=min(difference);
                temp(1,position)=temp(1,position)+0.6;
                difference(position)=9999;
                
                [~,position]=min(difference);
                temp(1,position)=temp(1,position)+0.3;
                difference(position)=9999;
                
%                 [~,position]=min(difference);
%                 temp(1,position)=temp(1,position)+0.15;
%                 difference(position)=9999;
                
                
            end
            temp=temp/sum(temp);   % normalization!
            
            if j<=train_sample+2
                train_total = train_total+1;
                data_train(train_total,:)=temp;
                
                label_train(train_total,:)=i-2;
%                 if i==4
%                     label_train(train_total,:)=i-3;
%                 end
            else
                test_total = test_total+1;
                data_test(test_total,:)=temp;
                label_test(test_total,:)=i-2;
%                 if i==4
%                     label_test(test_total,:)=i-3;
%                 end
            end
        end
    end
end

model = svmtrain(label_train,data_train,'-s 0 -t 2 -c 3.7 -g 40 -q');
[predictlabel,accuracy,~] = svmpredict(label_test,data_test,model)





