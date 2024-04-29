function [dataCube] = Load_and_Calibrate_Frame(calObject, fileNameStruct, frameNumber)

% Get File Names for the Master, Slave1, Slave2, Slave3
%fileNameStruct = getBinFileNames_withIdx(dataFolder, '0000');

% Load calibration file
load(calObject.calibrationfilePath);
RangeMat = calibResult.RangeMat;
PeakValMat = calibResult.PeakValMat;

%load calibration file
load(calObject.calibrationfilePath);
RangeMat = calibResult.RangeMat;
PeakValMat = calibResult.PeakValMat;

fileFolder = calObject.binfilePath;
frameIdx = calObject.frameIdx;

numSamplePerChirp = calObject.numSamplePerChirp;
nchirp_loops = calObject.nchirp_loops;
numChirpsPerFrame = calObject.numChirpsPerFrame;
TxToEnable = calObject.TxToEnable;
Slope_calib = calObject.Slope_calib;
fs_calib = calObject.fs_calib;
Sampling_Rate_sps = calObject.Sampling_Rate_sps;

chirpSlope = calObject.chirpSlope;
calibrationInterp = calObject.calibrationInterp;
TI_Cascade_RX_ID = calObject.TI_Cascade_RX_ID;
RxForMIMOProcess = calObject.RxForMIMOProcess;
IdTxForMIMOProcess = calObject.IdTxForMIMOProcess;
numRX = calObject.numRxToEnable;
phaseCalibOnly = calObject.phaseCalibOnly;
adcCalibrationOn = calObject.adcCalibrationOn;
N_TXForMIMO = calObject.N_TXForMIMO;
NumAnglesToSweep =  calObject.NumAnglesToSweep ;
RxOrder = calObject.RxOrder;
NumDevices = calObject.NumDevices;

numTX = length(TxToEnable);
outData = [];

numTX = length(TxToEnable);
%use the first TX as reference by default
TX_ref = TxToEnable(1);

% Get Valid Number of Frames
%[numValidFrames, ~] = getValidNumFrames(fullfile(dataFolder, fileNameStruct.masterIdxFile));
dataCube = [];
% Intentionally skip the first frame due to TDA2
%for frameIdx = 2:1:numValidFrames


numChirpPerLoop = numChirpsPerFrame/nchirp_loops;
numRXPerDevice = 4; % Fixed number

% ToDo: This is not efficient, it looks for eah frame. It
% should load everything at one
adcData = read_ADC_bin_TDA2_separateFiles(fileNameStruct,frameNumber,numSamplePerChirp,numChirpPerLoop,nchirp_loops, numRXPerDevice, 1);

for iTX = 1: numTX

    %use first enabled TX1/RX1 as reference for calibration
    TXind = TxToEnable(iTX);
    %       TXind = iTX;
    %construct the frequency compensation matrix
    freq_calib = (RangeMat(TXind,:)-RangeMat(TX_ref,1))*fs_calib/Sampling_Rate_sps *chirpSlope/Slope_calib;
    freq_calib = 2*pi*(freq_calib)/(numSamplePerChirp * calibrationInterp);
    correction_vec = (exp(1i*((0:numSamplePerChirp-1)'*freq_calib))');


    freq_correction_mat = repmat(correction_vec, 1, 1, nchirp_loops);
    freq_correction_mat = permute(freq_correction_mat, [2 3 1]);
    outData1TX = adcData(:,:,:,iTX).*freq_correction_mat;


    %construct the phase compensation matrix
    phase_calib = PeakValMat(TX_ref,1)./PeakValMat(TXind,:);
    %remove amplitude calibration
    if phaseCalibOnly == 1
        phase_calib = phase_calib./abs(phase_calib);
    end
    phase_correction_mat = repmat(phase_calib.', 1, numSamplePerChirp, nchirp_loops);
    phase_correction_mat = permute(phase_correction_mat, [2 3 1]);
    outData(:,:,:,iTX) = outData1TX.*phase_correction_mat;
end


% RX Channel re-ordering
outData = outData(:,:,RxForMIMOProcess,:);
%adcData = reshape(outData,size(outData,1), size(outData,2), size(outData,3)*size(outData,4));
adcData = outData;
dataCube = adcData;

end

