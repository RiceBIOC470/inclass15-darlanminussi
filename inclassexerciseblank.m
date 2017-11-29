%GB comments
Step1 0 No code or files of separately saved images
Step2 100
Step3 100 
Step4 100
Step5 100
Step6 100 
Step7 100 
Step8 100
Overall 88

%% step 1: write a few lines of code or use FIJI to separately save the
% nuclear channel of the image Colony1.tif for segmentation in Ilastik

% image provided was already only the DAPI channel

%% step 2: train a classifier on the nuclei
% try to get the get nuclei completely but separe them where you can
% save as both simple segmentation and probabilities

% done in ilastik

%% step 3: use h5read to read your Ilastik simple segmentation
% and display the binary masks produced by Ilastik 

% (datasetname = '/exported_data')
% Ilastik has the image transposed relative to matlab
% values are integers corresponding to segmentation classes you defined,
% figure out which value corresponds to nuclei

data_seg = h5read('Segmentation (Label 2).h5', '/exported_data');
data_seg = squeeze(data_seg);
imshow(data_seg, []);

% All the values that are different than zero will be the nuclei values in
% the segmentation data


% simple segmentation is the binary mask
% probability is the probability the probability that the pixel belongs to a certain class

%% step 3.1: show segmentation as overlay on raw data

orig_img = imread('48hColony1_DAPI.tif');

img_seg_overlay = cat(3, imadjust(orig_img), data_seg, zeros(size(orig_img)));
imshow(img_seg_overlay, []);

%% step 4: visualize the connected components using label2rgb
% probably a lot of nuclei will be connected into large objects

BW = im2bw(data_seg, graythresh(data_seg));
CC = bwconncomp(BW);

L = labelmatrix(CC);
rgb = label2rgb(L, 'jet', 'k');
imshow(rgb);


% Another way to visualize fused candidates without label2rgb
% separating fusioned candidates
stats = regionprops(CC, 'Area');
area = [stats.Area];
fusedCandidates = area > mean(area) + std(area);
sublist = CC.PixelIdxList(fusedCandidates);
sublist = cat(1, sublist{:});
fusedMask = false(size(data_seg));
fusedMask(sublist) = 1;
imshow(fusedMask, 'InitialMagnification', 'fit');

t = cat(3, fusedMask, BW, zeros(size(BW)));
imshow(t);




%% step 5: use h5read to read your Ilastik probabilities and visualize

% it will have a channel for each segmentation class you defined

data_prob = h5read('Prediction for Label 2.h5', '/exported_data');
data_prob = squeeze(data_prob);
imshow(data_prob, []);

%% step 6: threshold probabilities to separate nuclei better

data_prob_mask = data_prob > quantile(quantile(data_prob, 0.75),0.75);
imshow(data_prob_mask);

BW2 = im2bw(data_prob_mask);
CC2 = bwconncomp(BW2);

L2 = labelmatrix(CC2);
rgb2 = label2rgb(L2, 'jet', 'k');
imshow(rgb2);

img_merge = imfuse(data_prob, BW2);
imshow(img_merge);


%% step 7: watershed to fill in the original segmentation (~hysteresis threshold)

orig_img = imread('48hColony1_DAPI.tif');

img_mask = orig_img > quantile(quantile(orig_img, 0.75), 0.75);
imshow(img_mask, []);

CC = bwconncomp(img_mask);
stats = regionprops(CC, 'Area');
area = [stats.Area];
fusedCandidates = area > mean(area) + std(area);
sublist = CC.PixelIdxList(fusedCandidates);
sublist = cat(1, sublist{:});
fusedMask = false(size(img_mask));
fusedMask(sublist) = 1;
imshow(fusedMask, 'InitialMagnification', 'fit');

% eroding
s = round(1.2*sqrt(mean(area))/pi);
nucmin = imerode(fusedMask, strel('disk',s));
imshow(nucmin, 'InitialMagnification', 'fit');

% get outside region
outside = ~imdilate(fusedMask, strel('disk',1));
imshow(outside, 'InitialMagnification', 'fit');

% basins for ws
basin = imcomplement(bwdist(outside));
basin = imimposemin(basin, nucmin | outside);
pcolor(basin);
shading flat;

L = watershed(basin);
imshow(L);
colormap('jet');
caxis([0 20]);

% combining
newNuclearMask = L > 1 | (img_mask - fusedMask);
imshow(newNuclearMask, 'InitialMagnification', 'fit');

%% step 8: perform hysteresis thresholding in Ilastik and compare the results
% explain the differences

% the erosion method watershed done in matlab separates better larger
% fused nuclei but ends up removing nucleus smaller nuclei from the photo.

% the hysteresis filter in ilastik does a good job on separating nuclei, as
% seen in the attached exported image "final output,png" and is specially good in
% detecting nuclei on the outside area of the figure, however it is still
% possible to see some fused nuclei in the image

%% step 9: clean up the results more if you have time 
% using bwmorph, imopen, imclose etc

% cleans up small pixels
newNuclearMask_open = imopen(newNuclearMask, strel('disk', 5));
imshow(newNuclearMask_open,[]);

newNuclearMask_morph = bwmorph(newNuclearMask, 'remove');
imshow(newNuclearMask_morph);

newNuclearMask_close = imclose(newNuclearMask, strel('disk', 5));
imshow(newNuclearMask_open,[]);

