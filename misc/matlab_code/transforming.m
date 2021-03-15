%Load calibration parameters
load('stereparam.mat')

% Image 1 coordinates
% I1 = [[109, 328.62840536059656];  [108, 328.10704067640853]];
I1 = [[111.5963,  393.5897]; [330.5997,  149.0714]]; 

% Image 2 coordinates
% I2 = [[46, 33.3387896703047];  [46, 33.3387896703047]];
I2 = [[65.8821,   36.3804]; [611.2641,  271.3306]];

% find rotation, translation, and intrinsic params (note al of them are transposed)
R = stereoParams.RotationOfCamera2';
T = stereoParams.TranslationOfCamera2';
K = [stereoParams.CameraParameters2.IntrinsicMatrix' zeros(3,1)];

% looping through all possible pairs of I1 and I2 coordinates:
maxI1 = length(I1);
maxI2 = length(I2);

errors = containers.Map('KeyType','char', 'ValueType','any');
key_converter = @(tup) mat2str(tup);

for i=1:maxI1
    for j=1:maxI2
        point1 = I1(i,:);
        point2 = I2(j,:);
        
        % get world point using triangulation
        [point_world ,reprojectionErrors1] = triangulate(point1, point2, stereoParams);
        
        % reflect projection back to image 2 coordinates
        projection = K*[R T; 0 0 0 1]*[point_world'; 1];
        
        % make it homogenoous (last element must be 1)
        projected_I2 = projection/(projection(3));
        
        disp('projected:') 
        disp(projected_I2);
        disp('original:') 
        disp(point2);
        
        % calculate error of projection
        error = norm(projected_I2 - point2);
        errors(key_converter([i,j])) = error;
   end
end

disp("Error vals..");
disp(errors.values);

% for every I1 coor, choose I2 coor that minimises error (weighted graph
% problem)

% make it homogenoous (last element must be 1)
% % proj1 = projection/(projection(3));