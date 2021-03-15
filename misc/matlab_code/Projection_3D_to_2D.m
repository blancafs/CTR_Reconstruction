clear all
clc

%Load calibration parameters
load('stereparam.mat')


% Select a points from image captured by camera 1 (for instance the base or the tip)
% use right click to pick a point
img1 = imread('cam1_home_0.png');
% undistort the image
%img1 = undistortImage(img1,stereoParams.CameraParameters1);
imshow(img1)
[x1, y1] = getpts;
point1 = [x1, y1];


% Select same point from image captured by camera 2
img2 = imread('cam2_home_0.png');
%img2 = undistortImage(img2,stereoParams.CameraParameters2);
imshow(img2)
[x2, y2] = getpts;
point2 = [x2, y2];

close all

% Triangulation to find the corresponding world coordinate
[point_world ,reprojectionErrors1] = triangulate(point1, point2, stereoParams);


% Now we will project the 3D world coordinate on to image 2 

%Load the image from camera 2.
imOrig = imread('cam2_home_0.png');
% undistort the image
imUndistorted = undistortImage(imOrig,stereoParams.CameraParameters2);
imshow(imUndistorted);
hold on

% find rotation, translation, and intrinsic params (note al of them are transposed)
R = stereoParams.RotationOfCamera2';
T = stereoParams.TranslationOfCamera2';
K = [stereoParams.CameraParameters2.IntrinsicMatrix' zeros(3,1)];
projection = K*[R T; 0 0 0 1]*[point_world'; 1];
% make it homogenoous (last element must be 1)
proj1 = projection/(projection(3));
plot(proj1(1),proj1(2),'b*-');

%% you can also use the following function for 3D to 2D Projection
% project 3D points on to the image
%projectedPoints = worldToImage(stereoParams.CameraParameters2.Intrinsics,stereoParams.RotationOfCamera2,stereoParams.TranslationOfCamera2,point_world)
