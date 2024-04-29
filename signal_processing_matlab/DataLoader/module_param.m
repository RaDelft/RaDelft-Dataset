%  Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/ 
%  
%  
%   Redistribution and use in source and binary forms, with or without 
%   modification, are permitted provided that the following conditions 
%   are met:
%  
%     Redistributions of source code must retain the above copyright 
%     notice, this list of conditions and the following disclaimer.
%  
%     Redistributions in binary form must reproduce the above copyright
%     notice, this list of conditions and the following disclaimer in the 
%     documentation and/or other materials provided with the   
%     distribution.
%  
%     Neither the name of Texas Instruments Incorporated nor the names of
%     its contributors may be used to endorse or promote products derived
%     from this software without specific prior written permission.
%  
%   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
%   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
%   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
%   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
%   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
%   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
%   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
%   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
%   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
%   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
%   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%  
% 

% module_param.m
%  
% Contains a list of inital parameters for each modules used for signal
% processing, including modules of simTopCascade, calibration, rangeFFT, DopplerFFT, CFAR, DOA
% it is important to know that each parameter needs to be defined as
% moudleName_parameterName, the parameterName is defined in the
% corresponding module. Users need to know what parameters each module have
% before change this file.

platform = 'TI_4Chip_CASCADE';

%% fixed antenna ID and postion values for TI 4-chip cascade board. Should not be changed if user is based on TI board
% % Legacy Format TDA
% TI_Cascade_TX_position_azi = [0 4 8 12 16 20 24 28 32 9 10 11];%12 TX antenna azimuth position on TI 4-chip cascade EVM
% TI_Cascade_TX_position_ele = [0 0 0 0 0 0 0 0 0 1 4 6];%12 TX antenna elevation position on TI 4-chip cascade EVM

% New Format TDA
TI_Cascade_TX_position_azi = [11 10 9 32 28 24 20 16 12 8 4 0 ];%12 TX antenna azimuth position on TI 4-chip cascade EVM
TI_Cascade_TX_position_ele = [6 4 1 0 0 0 0 0 0 0 0 0];%12 TX antenna elevation position on TI 4-chip cascade EVM

% Legacy RX channel mapping
%TI_Cascade_RX_position_azi = [42:-1:39 3:-1:0 7:-1:4 53:-1:50 ];%16 RX antenna azimuth position on TI 4-chip cascade EVM
% TI_Cascade_RX_position_azi = [0:3 11:14 50:53 46:49 ];
TI_Cascade_RX_position_ele = zeros(1,16);%16 RX antenna elevation position on TI 4-chip cascade EVM
% TI_Cascade_RX_ID = [1 2 3 4 5 6 7 8 13 14 15 16 9 10 11 12]; %RX channel order on TI 4-chip cascade EVM

% Changed for the new Data format 
TI_Cascade_RX_position_azi = [ 11:14 50:53 46:49 0:3  ];
TI_Cascade_RX_ID = [13 14 15 16 1 2 3 4 9 10 11 12 5 6 7 8 ]; %RX channel order on TI 4-chip cascade EVM

TI_Cascade_Antenna_DesignFreq = 76.8; % antenna distance is designed for this frequency

%% constants
speedOfLight        = 3e8;
scaleFactor         = [0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125]*4;

%% define TX/RX antennas used for virtual array analysis. It can be a subset of the antennas enabled in data capture phase
%TxForMIMOProcess defines antenna TD used for TDM MIMO processing, can
%be a sub set of TxToEnable; CANNOT be channels that not enabled in TxToEnable
TxForMIMOProcess = TxToEnable;
%TxForMIMOProcess = [1:9  ];
[IdTxForMIMOProcess ia ib] = intersect(TxForMIMOProcess, TxToEnable,'stable' );
if length(IdTxForMIMOProcess)~= length(TxForMIMOProcess)
    error('TX channel used for processing is not valid')
    
end

RxForMIMOProcess = TI_Cascade_RX_ID; %using all 16 RXs, user can also choose subset of RXs for MIMO data analysis
%RxForMIMOProcess = [1:8];
D_TX = TI_Cascade_TX_position_azi(TxToEnable(ib)); %TX azimuth antenna coordinates
D_TX_ele = TI_Cascade_TX_position_ele(TxToEnable(ib));%TX elevation antenna coordinates

D_RX = TI_Cascade_RX_position_azi(RxForMIMOProcess); %RX azimuth antenna coordinate
D_RX_ele = TI_Cascade_RX_position_ele(RxForMIMOProcess);%RX elevation antenna coordinate

%draw the virtual array
plotArray = 0;
RX_id_tot = [];
RX_id_tot_ele = [];
if plotArray == 1
    figure(1);
end
for ii = 1:length(D_TX)
    RX_id_new = D_RX + sum(D_TX(ii));
    RX_id_tot = [RX_id_tot RX_id_new];
    
    RX_id_new_ele = D_RX_ele + D_TX_ele(ii);
    RX_id_tot_ele = [RX_id_tot_ele RX_id_new_ele];
    if plotArray == 1
        plot(RX_id_new,RX_id_new_ele,'o');grid on; hold on
        ylim([-8 8])
    end
end
D(:,1) = RX_id_tot;
D(:,2) = RX_id_tot_ele;


%% derived parameters
DopplerFFTSize = 2^(ceil(log2(nchirp_loops)));
numChirpsPerFrame = nchirp_loops*numChirpsInLoop;%nchirp_loops*numTxAnt;
chirpRampTime       = numADCSample/adcSampleRate;
chirpBandwidth      = chirpSlope * chirpRampTime; % Hz
chirpInterval       = chirpRampEndTime + chirpIdleTime;
carrierFrequency    = startFreqConst +  (adcStartTimeConst + chirpRampTime/2)*chirpSlope; % Hz center frequency
lambda              = speedOfLight/carrierFrequency;
maximumVelocity     = lambda / (chirpInterval*4) ; % m/s
maxRange            = speedOfLight*adcSampleRate*chirpRampTime/(2*chirpBandwidth);
numSamplePerChirp   = round(chirpRampTime*adcSampleRate);
rangeFFTSize        = 2^(ceil(log2(numSamplePerChirp)));
numChirpsPerVirAnt  = nchirp_loops;
numVirtualRxAnt     = length(TxForMIMOProcess) * length(RxForMIMOProcess) ;
rangeResolution     = speedOfLight/2/chirpBandwidth;
rangeBinSize        = rangeResolution*numSamplePerChirp/rangeFFTSize;
velocityResolution  = lambda / (2*nchirp_loops * chirpInterval*numTxAnt);
velocityBinSize     = velocityResolution*numChirpsPerVirAnt/DopplerFFTSize;


%% simTopCascade parameters
simTopCascade_inputDataSource          = 'bin'; % choose from 'mat','bin','gen'
simTopCascade_enable      = 1;
simTopCascade_outputDataSavingEnable   = 0;
simTopCascade_outputDataFileName       = [];
simTopCascade_totNumFrames = frameCount;
simTopCascade_platform                 = platform;


%% calibration cascade parameters
calibrationInterp              = 5;     %interpolation factor used for range FFT for frequency calibration, determined at calibration stage
calibrationCascade_enable      = 1;
calibrationCascade_binfilePath = [];
calibrationCascade_calibrationfilePath = [];
calibrationCascade_frameIdx = 1;
calibrationCascade_numSamplePerChirp = numSamplePerChirp;
calibrationCascade_nchirp_loops = nchirp_loops;
calibrationCascade_numChirpsPerFrame = numChirpsPerFrame;
calibrationCascade_TxToEnable = TxToEnable;
calibrationCascade_Slope_calib = Slope_calib;
calibrationCascade_Sampling_Rate_sps = adcSampleRate;
calibrationCascade_fs_calib = fs_calib;
calibrationCascade_chirpSlope = chirpSlope;
calibrationCascade_calibrationInterp = calibrationInterp;
calibrationCascade_TI_Cascade_RX_ID = TI_Cascade_RX_ID;
calibrationCascade_RxForMIMOProcess = RxForMIMOProcess;
calibrationCascade_IdTxForMIMOProcess = IdTxForMIMOProcess;
calibrationCascade_numRxToEnable = numRxToEnable;
calibrationCascade_phaseCalibOnly = 1; % 1: only phase calibration; 0: phase and amplitude calibration
calibrationCascade_adcCalibrationOn = 1; %1: adc data calibration on; 0 calibration off
calibrationCascade_ADVANCED_FRAME_CONFIG = ADVANCED_FRAME_CONFIG; %1 : indicate advance frame config, this value is passed from top main file
calibrationCascade_dataPlatform = dataPlatform;
calibrationCascade_RxOrder = TI_Cascade_RX_ID;
calibrationCascade_NumDevices = NumDevices;

if ADVANCED_FRAME_CONFIG == 1
    calibrationCascade_N_TXForMIMO = N_TXForMIMO; % used for read raw adc data in advanced frame config
    calibrationCascade_NumAnglesToSweep = NumAnglesToSweep;% used for read raw adc data in advanced frame config
end

%% range FFT parameters
rangeProcCascade_enable                = 1;
rangeProcCascade_numAntenna            = numVirtualRxAnt;      % number of antennas
rangeProcCascade_numAdcSamplePerChirp  = numSamplePerChirp;    % number of samples per chirp
rangeProcCascade_rangeFFTSize          = rangeFFTSize;         % FFT size
rangeProcCascade_dcOffsetCompEnable    = 1;
rangeProcCascade_rangeWindowEnable     = 1;                    % flag to enable or disable windowing before range FFT
% rangeProcCascade_rangeWindowCoeff      = [0.0800, 0.0894, 0.1173, 0.1624, 0.2231, 0.2967, 0.3802, 0.4703, 0.5633...
%                                     0.6553, 0.7426, 0.8216, 0.8890 0.9422, 0.9789, 0.9976]; % range FFT window coefficients
%windowCoeff = hann_local(numSamplePerChirp);
windowCoeff = hanning(numSamplePerChirp);
rangeProcCascade_rangeWindowCoeff      = windowCoeff(1:(numSamplePerChirp/2));
rangeProcCascade_scaleFactorRange      = scaleFactor(log2(rangeFFTSize) - 3);
rangeProcCascade_FFTOutScaleOn = 0; %1: apply scaleFactorRange; 0: scaling factor not applied

%% Doppler FFT parameters
DopplerProcClutterRemove_enable              = 1;
DopplerProcClutterRemove_numAntenna          = numVirtualRxAnt;      % number of antennas
DopplerProcClutterRemove_numDopplerLines     = rangeFFTSize;         % number of Doppler lines
DopplerProcClutterRemove_dopplerFFTSize      = DopplerFFTSize;       % Doppler FFT size
DopplerProcClutterRemove_numChirpsPerVirAnt  = numChirpsPerVirAnt;
DopplerProcClutterRemove_dopplerWindowEnable = 0;                    % flag to enable or disable windowing before Doppler FFT
windowCoeff = hanning(numChirpsPerVirAnt);
DopplerProcClutterRemove_dopplerWindowCoeff = windowCoeff(1:(numChirpsPerVirAnt/2));
DopplerProcClutterRemove_scaleFactorDoppler  = scaleFactor(log2(DopplerFFTSize) - 3);
DopplerProcClutterRemove_FFTOutScaleOn       = 0; %1: apply scaleFactorRange; 0: scaling factor not applied
DopplerProcClutterRemove_clutterRemove       = 0;                        %1=enable clutter removal; 0=no


%% detection parameters
ind = find(D(:,2)==0);
[val ID_unique] = unique(D(ind,1));
antenna_azimuthonly = ind(ID_unique); %virtual channel ID only for unique azimuth ID

CFAR_CASO_enable             = 1;
CFAR_CASO_detectMethod       = 1;                %(CASO-CFAR)dualpass rng/dop; only support one CFAR method
CFAR_CASO_numAntenna         = numVirtualRxAnt; %number of antennas
CFAR_CASO_refWinSize         = [8, 4];          % number of reference cells to estimate noise variance
CFAR_CASO_guardWinSize       = [8, 0];           % number of gap cells to prevent leakage being detected as signal
CFAR_CASO_K0                 = [5 3];       % Threshold scaling factor -- 6.3 corresponds to SNR of 8dB
CFAR_CASO_maxEnable          = 0;                %1: detect only if it is the maximum within window; 0: otherwise
CFAR_CASO_ratio_OS           = 0.65;             % percentage used to determine noise level used for detection
CFAR_CASO_rangeBinSize    = rangeBinSize;
CFAR_CASO_velocityBinSize  = velocityBinSize;
CFAR_CASO_dopplerFFTSize     = DopplerFFTSize;   % Doppler FFT size
CFAR_CASO_powerThre          = 0;                % if power of detected signal is less than this level, drop off this object.
CFAR_CASO_discardCellLeft    = 10;                % Number of range bins to discard due to distortions around DC (positive frequencies)
CFAR_CASO_discardCellRight   = 20;               % Number of range bins to discard due to distortions around DC (negative frequencies)
CFAR_CASO_numRxAnt           = length(RxForMIMOProcess);
CFAR_CASO_TDM_MIMO_numTX     = length(TxForMIMOProcess);
CFAR_CASO_antenna_azimuthonly = antenna_azimuthonly; %virtual channel ID only for unique azimuth ID
CFAR_CASO_minDisApplyVmaxExtend = 10; % meter, within this range, do not apply max velocity extension
CFAR_CASO_applyVmaxExtend = 0;

%find the overlap antenna ID that can be used for phase compensation
TX_ID_MIMO = repmat(1:length(TxForMIMOProcess),length(RxForMIMOProcess),1);
TX_ID_MIMO = TX_ID_MIMO(:);
sumTwo = D(:,1)*10 + D(:,2);
[val id] = unique(sumTwo);
id_repeat = setxor(id, 1:length(sumTwo));

overlapAntenna_ID = [];
%found the overlap antenna ID

for ii = 1:length(id_repeat)
    %ID of pair
    overlapAntenna_ID(ii,1:2) = find(sumTwo == sumTwo(id_repeat(ii)));
    %associated TX of each pair
    overlapAntenna_ID(ii,3:4) =TX_ID_MIMO(overlapAntenna_ID(ii,1:2));
end

if length(overlapAntenna_ID) > 0
    %find the pairs offset only by 1 chirp/TX in time
    dif_TX = abs(overlapAntenna_ID(:,3) - overlapAntenna_ID(:,4));
    ID_dif_1TX = find((dif_TX) == 1 );
    CFAR_CASO_overlapAntenna_ID = overlapAntenna_ID(ID_dif_1TX,:);
    ID_dif_2TX = find(dif_TX == 2);
    CFAR_CASO_overlapAntenna_ID_2TX = overlapAntenna_ID(ID_dif_2TX,:);
    ID_dif_3TX = find(dif_TX == 3);
    CFAR_CASO_overlapAntenna_ID_3TX = overlapAntenna_ID(ID_dif_3TX,:);
else
    CFAR_CASO_overlapAntenna_ID = [];
    CFAR_CASO_overlapAntenna_ID_2TX = [];
    CFAR_CASO_overlapAntenna_ID_3TX = [];
    
end


%% DOA parameters
% optimal d value used of for RF frequency of 76G 77G 78G 79G 80G
%d_optimal_calib = [0.495 0.504 0.51 0.516 0.522];
% switch centerFreq
%     case 76
%         d_optimal = d_optimal_calib(1);
%     case 77
%         d_optimal = d_optimal_calib(2);
%     case 78
%         d_optimal = d_optimal_calib(3);
%     case 79
%         d_optimal = d_optimal_calib(4);
%     case 80
%         d_optimal = d_optimal_calib(5);
% end

d_optimal = 0.5 * centerFreq / TI_Cascade_Antenna_DesignFreq;

DOACascade_enable                = 1;
DOACascade_D = D;
DOACascade_DOAFFTSize = 256;
DOACascade_numAntenna          = numVirtualRxAnt;
DOACascade_antPos          = [0:numVirtualRxAnt-1];
DOACascade_antDis              = d_optimal;              % in terms of lamda
DOACascade_method              = 1;                % 1: 2D  muli-object beamforming, 2: 2D  muli-object beamforming and peak search after azi/ele FFT
DOACascade_angles_DOA_az=[-80 80]; %field of view to run 2D DOA in azimuth
DOACascade_angles_DOA_ele = [-20 20];%field of view to run 2D DOA in elevation
DOACascade_gamma  = 10^(0.2/10);      % Used in peak detection
DOACascade_sidelobeLevel_dB_azim = 1; % used to reject sidelobe in azimuth run 2D DOA
DOACascade_sidelobeLevel_dB_elev = 0;% used to reject sidelobe in elevation run 2D DOA
DOACascade_dopplerFFTSize      = DopplerFFTSize;


