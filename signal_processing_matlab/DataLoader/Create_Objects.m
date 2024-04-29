function [radarObject, calibrationObject] = Create_Objects(dataFolder)

% Constant paths
%calFile = 'calibrateResults_football.mat';
calFile = 'calibrateResults_sen_calib.mat';
modParamFile = 'module_param.m';
pathGenParaFile = [dataFolder,'params.m'];
dataPlatform = 'TDA2';

% Save parameters
parameter_file_gen_json(dataFolder, calFile, modParamFile, pathGenParaFile, dataPlatform);

 % Load parameters
run(pathGenParaFile);
            
% Create CalibrationObject
calibrationObject = calibrationCascade(pathGenParaFile);
calibrationObject.calibrationfilePath = calFile;

% Create RadarObject
radarObject.calibrationfilePath = calFile;
radarObject.rangeBinSize = rangeBinSize;            
radarObject.maxRange = maxRange;
radarObject.velocityBinSize = velocityBinSize;
radarObject.maxVelocity = maximumVelocity;
radarObject.doaAngles = DOACascade_angles_DOA_az;
radarObject.lambda = lambda;
radarObject.radarObject.fs = adcSampleRate;
radarObject.radarObject.adcStartTime = adcStartTimeConst;
radarObject.radarObject.centerFreq = centerFreq;
radarObject.radarObject.bandWidth = chirpBandwidth;
radarObject.radarObject.chirpSlope = chirpSlope;
radarObject.radarObject.chirpEnd = chirpRampEndTime;
radarObject.radarObject.chirpInterval = chirpInterval;
radarObject.radarObject.chirpIddle = chirpIdleTime;
radarObject.radarObject.TxToEnable = TxToEnable;
radarObject.radarObject.n_angle_fft_size = 256;
radarObject.radarObject.numChirpsPerFrame = numChirpsPerFrame;
radarObject.nchirp_loops = nchirp_loops;
radarObject.numSamplePerChirp = numSamplePerChirp;
radarObject.antenna_azimuthonly = antenna_azimuthonly;
radarObject.n_angle_fft_size = 256;
            
            
end

