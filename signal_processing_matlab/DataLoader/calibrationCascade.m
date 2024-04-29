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

%calibrationCascade.m
%
% calibrationCascade module definition. This module calibrates the ADC data with the calibration
% matrix installed with the path name given by calibrationfilePath.
% Calibration is done directly on the raw ADC data before any further
% processing. Apply frequency and phase calibration in time domain; amplitude
% calibration is optional, can be turned on or off


%% Class definition
classdef calibrationCascade
    
    %% properties
    properties (Access = public)
        %method 
       name = 'calibrationCascade';
       pfile = [];
       binfilePath = []   
       calibrationfilePath = []   
       frameIdx = 1
       adcCalibrationOn = 1;  
       
       %chirpParameters
       numSamplePerChirp = 0
       nchirp_loops = 0
       numChirpsPerFrame = 0
       TxToEnable = []
       Slope_calib = 0
       Sampling_Rate_sps = 0;
       fs_calib = 0;
       chirpSlope = 0
       calibrationInterp = 0
       TI_Cascade_RX_ID = []
       RxForMIMOProcess = []
       IdTxForMIMOProcess = []
       numRxToEnable = 0
       phaseCalibOnly = 0
       ADVANCED_FRAME_CONFIG = 0
       N_TXForMIMO = 0
       NumAnglesToSweep = 0
       dataPlatform = [];
       RxOrder = [];
       NumDevices = 4;
       
    end
    
    methods
        
        %% constructor
        function obj = calibrationCascade(pathGenParaFile)
            
            run(pathGenParaFile);
            
            obj.pfile = pathGenParaFile;
            % Set parameters
            obj.binfilePath = eval(strcat(obj.name,'_','binfilePath'));       
            obj.frameIdx = eval(strcat(obj.name,'_','frameIdx'));           
            obj.numSamplePerChirp = eval(strcat(obj.name,'_','numSamplePerChirp'));
            obj.nchirp_loops = eval(strcat(obj.name,'_','nchirp_loops'));
            obj.numChirpsPerFrame = eval(strcat(obj.name,'_','numChirpsPerFrame'));
            obj.TxToEnable = eval(strcat(obj.name,'_','TxToEnable'));
            obj.Slope_calib = eval(strcat(obj.name,'_','Slope_calib'));
            obj.Sampling_Rate_sps = eval(strcat(obj.name,'_','Sampling_Rate_sps'));
            obj.fs_calib = eval(strcat(obj.name,'_','fs_calib'));
            obj.chirpSlope = eval(strcat(obj.name,'_','chirpSlope'));
            obj.calibrationInterp = eval(strcat(obj.name,'_','calibrationInterp'));
            obj.TI_Cascade_RX_ID = eval(strcat(obj.name,'_','TI_Cascade_RX_ID'));
            obj.RxForMIMOProcess = eval(strcat(obj.name,'_','RxForMIMOProcess'));
            obj.IdTxForMIMOProcess = eval(strcat(obj.name,'_','IdTxForMIMOProcess'));
            obj.numRxToEnable = eval(strcat(obj.name,'_','numRxToEnable'));  
            obj.phaseCalibOnly = eval(strcat(obj.name,'_','phaseCalibOnly'));    
            obj.adcCalibrationOn = eval(strcat(obj.name,'_','adcCalibrationOn'));
            obj.ADVANCED_FRAME_CONFIG = eval(strcat(obj.name,'_','ADVANCED_FRAME_CONFIG'));
            
            if obj.ADVANCED_FRAME_CONFIG  == 1
                obj.N_TXForMIMO = eval(strcat(obj.name,'_','N_TXForMIMO'));
                obj.NumAnglesToSweep = eval(strcat(obj.name,'_','NumAnglesToSweep'));
            end
            
            obj.dataPlatform = eval(strcat(obj.name,'_','dataPlatform'));
            obj.RxOrder = eval(strcat(obj.name,'_','RxOrder'));
            obj.NumDevices = eval(strcat(obj.name,'_','NumDevices'));            
            
        end
    end
    
end

