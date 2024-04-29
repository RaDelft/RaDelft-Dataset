clear
addpath("DataLoader/")
addpath("DataLoader/JsonParser/")
addpath("Utils/")
%% INIT
% Config
TDMAmotionCompensation = 1;
saveDopplerFold = 0;
rangeFFTSize = 512;
dopplerFFTSize = 128;
angleFFTSize = 256;
elevationFFTsize = 128;

% Path to the raw data
% The current implementation process the scenes one by one.
% Change the path if you want to process other scene
rootFolder = 'PATH_TO_DATA/Day2Experiment1/';
dataFolder = strcat(rootFolder, 'RawData/');

% Path to save the radar cube
rcFolder = strcat(rootFolder, 'RadarCubes/');
if ~exist(rcFolder, 'dir')
    mkdir(rcFolder)
end

% Load the objects
[radarObject, calibrationObject] = Create_Objects(dataFolder);
antenna_azimuthonly = radarObject.antenna_azimuthonly;

% Load the antenna positions
D = load('D.mat').D;
D = D + 1;

globalFrameCounter = 1;

% Get Unique File Idxs in the "dataFolder_test"
[fileIdx_unique] = getUniqueFileIdx(dataFolder);
for i_file = 1:(length(fileIdx_unique))


    % Get File Names for the Master, Slave1, Slave2, Slave3
    fileNameStruct = getBinFileNames_withIdx(dataFolder, fileIdx_unique{i_file});

    % Get Valid Number of Frames
    [numValidFrames, ~] = getValidNumFrames(fullfile(dataFolder, fileNameStruct.masterIdxFile));

    disp('----------------------')
    fprintf('START PROCESSING FILES: %s \n', fileIdx_unique{i_file})
    fprintf('TOTAL NUMBER OF FRAMES: %d \n', numValidFrames-1)
    disp('----------------------')

    % Process Data Frame by Frame
    % Intentionally skip the first frame due to TDA2
    for nFrame= 2:numValidFrames

        fprintf('START FRAME PROCESSING: %d \n', nFrame)

        % Load one frame of raw data
        rawData = Load_and_Calibrate_Frame(calibrationObject, fileNameStruct, nFrame);

        % Reshape in virtual channels
        rawData = reshape(rawData,size(rawData,1), size(rawData,2), size(rawData,3)*size(rawData,4));

        % Range Processing
        [rawData, ~] = windowing(rawData,1,6);
        rangeFFTData = fft(rawData, rangeFFTSize, 1) / rangeFFTSize;

        % Doppler Processing
        [rangeFFTData, ~] = windowing(rangeFFTData,2,6);
        rangeDopplerFFTData = fft(rangeFFTData, dopplerFFTSize, 2) / dopplerFFTSize;
        rangeDopplerFFTData = fftshift(rangeDopplerFFTData,2);
        
        % Remove first and last two cells
        rangeDopplerFFTData = rangeDopplerFFTData(11:end-2,:,:);

        % TDMA compensation
        [rangeDopplerFFTData, dopplerFold] = TDMAPhaseCorrectionVector(rangeDopplerFFTData);
        
        
        % Arrange array topology and zero filling       
        apertureLen_azim = max(D(:,1));
        apertureLen_elev = max(D(:,2));
        radarCube = zeros(rangeFFTSize-12, dopplerFFTSize,apertureLen_azim,apertureLen_elev,'single');
        for i_line = 1:apertureLen_elev
            indice = find(D(:,2) == i_line);
            D_sel = D(indice,1);
            sig_sel = rangeDopplerFFTData(:,:,indice);
            [val, indU] = unique(D_sel);

            radarCube(:,:,D_sel(indU),i_line) = sig_sel(:,:,indU);
        end

        % Azimuth FFT
        radarCube = fft(radarCube,angleFFTSize,3);       
        radarCube = fftshift(radarCube,3);

        % This is to get +-70 degrees in azimuth. 
        % Only valid with the current FFT size
        radarCube = radarCube(:,:,9:248,:);
        
        % Elevation FFT
        radarCube = fftshift(fft(radarCube,elevationFFTsize,4),4);
        
        % This is to get +-20 degrees. Only valid with the current FFT size
        % Only valid with the current FFT size
        radarCube = radarCube(:,:,:,43:86);
        radarCube = abs(radarCube);        

        % Find the peak in elevation and save the index
        [~,elevationIndex] = max(radarCube,[],4);
        elevationIndex = uint8(elevationIndex);

        % We repeat the azimuth FFT. This is not optimal, but we want to 
        % apply a window function now. In the URA processing is not
        % possible to apply a window in the x-direction because it is very
        % sparse
        radarCube = rangeDopplerFFTData(:,:,antenna_azimuthonly);
        radarCube = windowing(radarCube,3,6);
        radarCube = fft(radarCube,angleFFTSize,3);
        radarCube = fftshift(radarCube,3);
        
        % This is to get +-70 degrees in azimuth. 
        % Only valid with the current FFT size
        radarCube= abs(radarCube(:,:,9:248));
        radarCube = single(radarCube);

        % Save Radar Cube
        fileName = strcat(rcFolder,'Pow_Frame_',num2str(globalFrameCounter + nFrame -2),'.mat');
        fileNameElevation = strcat(rcFolder,'Ele_Frame_',num2str(globalFrameCounter + nFrame -2),'.mat');
        save(fileName,'radarCube')
        save(fileNameElevation,'elevationIndex')
        
        % If saveDopplerFold is active, save the Doppler index with the
        % unambiguous region exteded. By default not save it because it
        % takes too much space.
        if saveDopplerFold
            fileNameDopplerFold = strcat(rcFolder,'DopFold_Frame_',num2str(globalFrameCounter + nFrame -2),'.mat');
            save(fileNameDopplerFold, 'dopplerFold')
        end

    end

    globalFrameCounter = globalFrameCounter + numValidFrames - 1;
end