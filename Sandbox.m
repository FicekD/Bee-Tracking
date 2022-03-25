%% Uklid
clc;
clear all;
close all;

%% Get the single ROIs
tunnels = struct;
background = im2double(rgb2gray(imread('data\210906_Pokus2\Image0014 11-39-24.jpg')));
vcelky = im2double(rgb2gray(imread('data\210906_Pokus2_sorted\Image1399 11-44-01.jpg')));

% Define the tunnel coordinates array by hand
tunnels(1).tCoordB = 95;
tunnels(1).tCoordE = 150;

tunnels(2).tCoordB = 170;
tunnels(2).tCoordE = 220;

tunnels(3).tCoordB = 245;
tunnels(3).tCoordE = 300;

tunnels(4).tCoordB = 320;
tunnels(4).tCoordE = 390;

tunnels(5).tCoordB = 400;
tunnels(5).tCoordE = 475;

tunnels(6).tCoordB = 485;
tunnels(6).tCoordE = 560;

tunnels(7).tCoordB = 570;
tunnels(7).tCoordE = 650;

tunnels(8).tCoordB = 655;
tunnels(8).tCoordE = 730;

tunnels(9).tCoordB = 745;
tunnels(9).tCoordE = 810;

tunnels(10).tCoordB = 825;
tunnels(10).tCoordE = 885;

tunnels(11).tCoordB = 910;
tunnels(11).tCoordE = 960;

tunnels(12).tCoordB = 985;
tunnels(12).tCoordE = 1030;

% Get the fine ROIs
for i = 1:12
    % Prepare the background values
    imgTempB = background(:, tunnels(i).tCoordB:tunnels(i).tCoordE);

    tunnels(i).levelBckg = graythresh(imgTempB);
    tunnels(i).mask = imbinarize(imgTempB, tunnels(i).levelBckg);
    tunnels(i).fillRatioBckg = 1 - nnz(tunnels(i).mask)/(size(imgTempB, 1) * size(imgTempB, 2));

    figure(1)
    subplot(1, 12, i)
    imshow(tunnels(i).mask)

    % Process the bee images
    imgTempV = vcelky(:, tunnels(i).tCoordB:tunnels(i).tCoordE);

    maskTemp = imbinarize(imgTempV, tunnels(i).levelBckg);
    tunnels(i).fillRatioVc = 1 - nnz(maskTemp)/(size(imgTempB, 1) * size(imgTempB, 2));

    tunnels(i).occRatio = tunnels(i).fillRatioVc / tunnels(i).fillRatioBckg;

    figure(2)
    subplot(1, 12, i)
    imshow(maskTemp)
end

%% Pokusy
figure(3)
subplot(1, 4, 1)
imshow(tunnels(i).mask)
subplot(1, 4, 2)
imshow(maskTemp)
subplot(1, 4, 3)
imshow(tunnels(i).mask - maskTemp)
subplot(1, 4, 4)
imshow(maskTemp - tunnels(i).mask)