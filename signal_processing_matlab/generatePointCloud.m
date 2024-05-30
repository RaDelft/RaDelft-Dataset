clear
addpath("Detectors/")
% Range Axis
% This is only valid for the specific BW and zero-padding we used. 
% It has to be changed if the proccesing or the BW is changed.
rangeCellSize = 0.1004;
rangeAxis = rangeCellSize:rangeCellSize :51.4242;
rangeAxis = rangeAxis(11:end-2);

% Azimuth Axis
angleFFTSize = 256;
wx_vec=-pi:2*pi/(angleFFTSize-1):pi;
wx_vec = flip(wx_vec);
wx_vec = wx_vec(9:248);
azimuthAxis = asin(wx_vec/(2*pi*0.4972));

% Elevation Axis
eleFFTSize = 128;
wz_vec=-pi:2*pi/(eleFFTSize-1):pi;
wz_vec = flip(wz_vec);
wz_vec = wz_vec(43:86);
elevationAxis = asin(wz_vec/(2*pi*0.4972));

% Paths
% The current implementation process the captures scene by scene
folder = '/PATH_TO_DATA/Scene2/RadarCubes/';
files = dir(fullfile(folder,'Pow_Frame*.mat'));

active2D = 1; % 1 means 2D version, all elevation=0; 0 means 3D, use elevation

% First load all the data, frame by frame, then plot it.
parfor frame=1:size(files,1)
    fprintf('START FRAME PROCESSING: %d \n', frame)

    % Load Data
    fileNamePow = strcat(folder, 'Pow_Frame_', num2str(frame),'.mat');
    radarCube = load(fileNamePow).radarCube;

    if ~active2D
        fileNameEle = strcat(folder, 'Ele_Frame_', num2str(frame),'.mat');        
        eleCube = load(fileNameEle).elevationIndex;
    end

    % Run detector
    [rangeBins, dopplerBins, azimuthBins] = detOSOSOS2(radarCube);
    
    rangeBins = reshape(rangeBins, [], 1);
    dopplerBins = reshape(dopplerBins, [], 1);
    azimuthBins = reshape(azimuthBins, [], 1);

    % Get polar
    range = rangeAxis(rangeBins);
    azimuth = azimuthAxis(azimuthBins);
    indices = sub2ind(size(radarCube), rangeBins, dopplerBins, azimuthBins);

    power = mag2db(radarCube(indices)).';
    
    % Compute elevation if needed.
    if active2D
        elevation = zeros(size(range));
    else
        elevation = elevationAxis(eleCube(indices));
    end
 
    % Convert to Cartesian coordinates
    [x,y,z] = sph2cart(azimuth,elevation,range);

    pointCloud{frame} = [x;y;z;power];
            
end
%% ROS DS Generation
load(strcat(folder,'timestamps.mat'));
saveFolder ='/PATH_TO_SAVE_DIR/Scene2/rosDS/radar_ososos2D/';


if ~exist(saveFolder, 'dir')
       mkdir(saveFolder);
end

% Add Radar Point Cloud to the new files
for i=1:min(size(pointCloud,2), size(unixDateTime,1))
    timeStamp = unixDateTime(i);
    rosTimeStamp = rostime(timeStamp);
    timeStampString = strcat(num2str(rosTimeStamp.Sec, '%012d'),'.',num2str(rosTimeStamp.Nsec,'%.9d'));
    savePath =strcat(saveFolder, timeStampString, '.mat');
    points = pointCloud{i};

    
    if size(points,1)>0
        points = points.';
        points = cast(points,"single");
        save(savePath, 'points')
    end

end

%% ROS Bag Generation
% Activate only if you want the point cloud in rosbag format
%{
% Load timestamps
load(strcat(folder,'timestamps.mat'));


% Create new bag
path = strcat(folder,'OSOSPeak6.bag');
bagwriter = rosbagwriter(path);


% Add Radar Point Cloud to the new bag
for i=1:min(size(pointCloud,2), size(unixDateTime,1))

    timeStamp = unixDateTime(i);
    points = pointCloud{i};
    if size(points,1)>0
        points = points.';
        Doppler = points(:,4);
        points = points(:,1:3);
        points = cast(points,"single");
        Doppler = cast(Doppler, 'single');
        lidarMsgOut = packagePointCloud(points,Doppler,timeStamp);

        write(bagwriter,"/radar_osospeak6",timeStamp,lidarMsgOut);

    end

end
%}
