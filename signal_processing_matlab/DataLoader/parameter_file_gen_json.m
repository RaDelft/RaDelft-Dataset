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

% parameter_file_gen_Jason.m
%
% Function call to generate a complete parameter file using Jason file before signal processing, including chirp parameters and module parameters.
% This function does not support advanced frame config
%input:
%   test_name: test data file name which contains the chirp configuration
%   file in a Jason file format
%   dataFolder_calib: calibration data file path which contains the pre-calibrated .mat file. This .mat file includes the calibration matrix to be used in adc data calibration module
%   module_param_file: contains parameters to initialize each signal processing module
%   pathGenParaFile: the complete parameter file name to be generated
%   dataPlatform: flag for data capture platform


function parameter_file_gen_json(test_name, dataFolder_calib,module_param_file, pathGenParaFile, dataPlatform)

% create param file
%param_fileName = strcat(test_name,'test_param.m');
fidParam = fopen(pathGenParaFile, 'w');

fprintf(fidParam, 'ADVANCED_FRAME_CONFIG = 0; \n');
fprintf(fidParam, 'dataPlatform = ''%s''; \n', dataPlatform);

%% find *.json file in the same folder of the test data; if json files in the folder, this code will NOT work, please make sure only one exist
f=dir(fullfile(test_name,'*.mmwave.json'));
if length(f)~=1
    error('Unknown parameter file or too many chirpProfiles_*.json file!!');
else
    
    %disp(['paramFile= ' test_name f.name]);
    paramFile = [test_name f.name];
    
    %go to the test data folder to run the .json file and read the chirp parameters
    %associated with the data
    params_chirp = JsonParser(paramFile);
    
    numChirpConfig = length(params_chirp.DevConfig(1).Chirp);
    numTXPerDev = 3;
    totTx = numTXPerDev*params_chirp.NumDevices;
    TxEnableTable = zeros(numChirpConfig, totTx);
    for iDev = 1:params_chirp.NumDevices
        for iconfig = 1:numChirpConfig
            TxEnableTable(iconfig,1+(iDev-1)*numTXPerDev) = ...
                params_chirp.DevConfig(iDev).Chirp(iconfig).Tx0Enable;
            TxEnableTable(iconfig, 2+(iDev-1)*numTXPerDev) = ...
                params_chirp.DevConfig(iDev).Chirp(iconfig).Tx1Enable;
            TxEnableTable(iconfig, 3+(iDev-1)*numTXPerDev) = ...
                params_chirp.DevConfig(iDev).Chirp(iconfig).Tx2Enable;
        end
    end
    
    TxChannelEnabled = zeros(1,numChirpConfig);
    for iconfig = 1:numChirpConfig
        [channelID] = find(TxEnableTable(iconfig,:)~=0);
        TxChannelEnabled(iconfig) = channelID;
    end
    
    %% pass the chirp parameter file to _param.m
    % assuming 4 devices use the same config, so only device 1 chirp configuration information
    % is passed to the post processing
    fprintf(fidParam, '%%pass the chirp parameters associated with test data \n');
    fprintf(fidParam, 'numADCSample = %e; \n', params_chirp.DevConfig(1).Profile.NumSamples);
    fprintf(fidParam, 'adcSampleRate = %e; %%Hz/s \n', params_chirp.DevConfig(1).Profile.SamplingRate*1e3);
    fprintf(fidParam, 'startFreqConst = %e; %%Hz \n', params_chirp.DevConfig(1).Profile.StartFreq*1e9);
    fprintf(fidParam, 'chirpSlope = %e; %%Hz/s \n',params_chirp.DevConfig(1).Profile.FreqSlope*1e12);
    fprintf(fidParam, 'chirpIdleTime = %e; %%s \n', params_chirp.DevConfig(1).Profile.IdleTime*1e-6);
    fprintf(fidParam, 'adcStartTimeConst = %e; %%s \n', params_chirp.DevConfig(1).Profile.AdcStartTime*1e-6);
    fprintf(fidParam, 'chirpRampEndTime = %d; %%s \n', params_chirp.DevConfig(1).Profile.RampEndTime*1e-6);
    fprintf(fidParam, 'framePeriodicty = %d; \n', params_chirp.DevConfig(1).FrameConfig.Periodicity * 1e-3);
    fprintf(fidParam, 'NumDevices = %d; \n', params_chirp.NumDevices);
    fprintf(fidParam, 'framePeriodicty = %d; \n', params_chirp.DevConfig(1).FrameConfig.Periodicity * 1e-3);
    
    fprintf(fidParam, 'frameCount = %e; %%s \n', params_chirp.DevConfig(1).FrameConfig.NumFrames);
    fprintf(fidParam, 'numChirpsInLoop = %e; %%s \n', params_chirp.DevConfig(1).NumChirps);
    
    
    fprintf(fidParam, 'nchirp_loops = %d; \n', params_chirp.DevConfig(1).FrameConfig.NumChirpLoops);
    
    fprintf(fidParam, 'numTxAnt = %d; \n', length(params_chirp.TxToEnable));
    fprintf(fidParam, ['TxToEnable = [' num2str(TxChannelEnabled) ']' ';\n']);
    fprintf(fidParam, 'numRxToEnable = %d; \n', length(params_chirp.RxToEnable));
    
    Start_Freq_GHz = params_chirp.DevConfig(1).Profile.StartFreq;
    Samples_per_Chirp = params_chirp.DevConfig(1).Profile.NumSamples;
    Sampling_Rate_ksps = params_chirp.DevConfig(1).Profile.SamplingRate;
    Slope_MHzperus = params_chirp.DevConfig(1).Profile.FreqSlope;
    fprintf(fidParam, 'centerFreq = %d; \n', ((Start_Freq_GHz + Samples_per_Chirp/...
        Sampling_Rate_ksps*Slope_MHzperus/2)));
    
    
    %% calibration cascade parameters
    load(dataFolder_calib)
    paramsCalib = params;
    clear params;
    fprintf(fidParam, '%%pass the slope used for calibration \n');
    fprintf(fidParam, 'Slope_calib = %d; \n\n', paramsCalib.Slope_MHzperus*1e12);
    fprintf(fidParam, 'fs_calib = %d; \n\n', paramsCalib.Sampling_Rate_sps);
    
    %% combine with the rest of the signal
    fprintf(fidParam, '%%pass all other parameters \n');
    fidCommon = fopen(module_param_file);
    for ii = 1:32
        tline = fgets(fidCommon);
    end
    tline = fgets(fidCommon);
    while ischar(tline)
        fwrite(fidParam, tline);
        tline = fgets(fidCommon);
    end
    fclose(fidParam);
    fclose(fidCommon);
    
end


