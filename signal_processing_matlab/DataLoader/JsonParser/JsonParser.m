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

function [Params] = JsonParser(MmWaveJsonFile)
	
	%read the json file into structure
    %mmwaveJSON = jsondecode(fileread(mmWaveJsonFile));
	%MmwaveJSON = parse_json(fileread(MmWaveJsonFile));
	MmwaveJSON = loadjson(MmWaveJsonFile);
	MmWaveDevicesConfig = MmwaveJSON.mmWaveDevices;
    
    % onec device corresponding to each element in mmWaveDevices
    Params.NumDevices = length(MmWaveDevicesConfig);
    
    % Get the Tx/Rx channels enabled
	TxToEnable = [];
    RxToEnable = [];
    DevIdMap = [];
    
    for Count = 1:Params.NumDevices
        %get the device id corresponding to this config
		DevIdMap(Count) = MmWaveDevicesConfig{Count}.mmWaveDeviceId + 1;
		
        DevId = DevIdMap(Count);
        
	    if (strcmp(MmWaveDevicesConfig{Count}.rfConfig.waveformType, 'singleFrameChirp'))
            Params.FrameType = 0;
		else
			if (strcmp(MmWaveDevicesConfig{Count}.rfConfig.waveformType, 'advancedFrameChirp'))
				Params.FrameType = 1;
			else
				if (strcmp(MmWaveDevicesConfig{Count}.rfConfig.waveformType, 'continuousWave'))
					Params.FrameType = 2;
				end
			end
		end
		
		%get the tx channel enable
		TxChannelEn = hex2dec(MmWaveDevicesConfig{Count}.rfConfig.rlChanCfg_t.txChannelEn(3:end));
		for TxChannel = 1:3
			if(bitget(TxChannelEn, TxChannel))
				TxToEnable = [TxToEnable (3*(DevIdMap(Count)-1))+TxChannel];
			end
        end
        
        %get the rx channel enable
        RxChannelEn = hex2dec(MmWaveDevicesConfig{Count}.rfConfig.rlChanCfg_t.rxChannelEn(3:end));
		for RxChannel = 1:4
			if(bitget(RxChannelEn, RxChannel))
				RxToEnable = [RxToEnable (4*(DevIdMap(Count)-1))+RxChannel];
			end
        end
		
        Params.DevConfig(DevId).NumProfiles = length(MmWaveDevicesConfig{Count}.rfConfig.rlProfiles);
        NumChirpBlocks = length(MmWaveDevicesConfig{Count}.rfConfig.rlChirps);
        Profiles = MmWaveDevicesConfig{Count}.rfConfig.rlProfiles;
        Chirps = MmWaveDevicesConfig{Count}.rfConfig.rlChirps;
        
        % Profile Configuration
        for ProfileCount = 1:Params.DevConfig(DevId).NumProfiles
            Params.DevConfig(DevId).Profile(ProfileCount).ProfileId		 		= Profiles{ProfileCount}.rlProfileCfg_t.profileId;
            Params.DevConfig(DevId).Profile(ProfileCount).StartFreq		 		= Profiles{ProfileCount}.rlProfileCfg_t.startFreqConst_GHz;
            Params.DevConfig(DevId).Profile(ProfileCount).FreqSlope		 		= Profiles{ProfileCount}.rlProfileCfg_t.freqSlopeConst_MHz_usec;
            Params.DevConfig(DevId).Profile(ProfileCount).IdleTime		 		= Profiles{ProfileCount}.rlProfileCfg_t.idleTimeConst_usec;
            Params.DevConfig(DevId).Profile(ProfileCount).AdcStartTime	 		= Profiles{ProfileCount}.rlProfileCfg_t.adcStartTimeConst_usec;
            Params.DevConfig(DevId).Profile(ProfileCount).RampEndTime	 		= Profiles{ProfileCount}.rlProfileCfg_t.rampEndTime_usec;
            Params.DevConfig(DevId).Profile(ProfileCount).TxStartTime	 		= Profiles{ProfileCount}.rlProfileCfg_t.txStartTime_usec;
            Params.DevConfig(DevId).Profile(ProfileCount).RxGain		 		= hex2dec(Profiles{ProfileCount}.rlProfileCfg_t.rxGain_dB(3:end));
            Params.DevConfig(DevId).Profile(ProfileCount).NumSamples	 		= Profiles{ProfileCount}.rlProfileCfg_t.numAdcSamples;
            Params.DevConfig(DevId).Profile(ProfileCount).SamplingRate	 		= Profiles{ProfileCount}.rlProfileCfg_t.digOutSampleRate;
			Params.DevConfig(DevId).Profile(ProfileCount).HpfCornerFreq1	    = Profiles{ProfileCount}.rlProfileCfg_t.hpfCornerFreq1;
			Params.DevConfig(DevId).Profile(ProfileCount).HpfCornerFreq2	    = Profiles{ProfileCount}.rlProfileCfg_t.hpfCornerFreq2;
			Params.DevConfig(DevId).Profile(ProfileCount).Tx0PhaseShift	 		= (bitand(hex2dec(Profiles{ProfileCount}.rlProfileCfg_t.txPhaseShifter(3:end)), 255)/4)*5.625;
            Params.DevConfig(DevId).Profile(ProfileCount).Tx1PhaseShift	 		= (bitand(bitshift(hex2dec(Profiles{ProfileCount}.rlProfileCfg_t.txPhaseShifter(3:end)), -8), 255)/4)*5.625;
            Params.DevConfig(DevId).Profile(ProfileCount).Tx2PhaseShift	 		= (bitand(bitshift(hex2dec(Profiles{ProfileCount}.rlProfileCfg_t.txPhaseShifter(3:end)), -16), 255)/4)*5.625;
            Params.DevConfig(DevId).Profile(ProfileCount).Tx0OutPowerBackOff	= bitand(hex2dec(Profiles{ProfileCount}.rlProfileCfg_t.txOutPowerBackoffCode(3:end)), 255);
            Params.DevConfig(DevId).Profile(ProfileCount).Tx1OutPowerBackOff	= bitand(bitshift(hex2dec(Profiles{ProfileCount}.rlProfileCfg_t.txOutPowerBackoffCode(3:end)), -8), 255);
			Params.DevConfig(DevId).Profile(ProfileCount).Tx2OutPowerBackOff	= bitand(bitshift(hex2dec(Profiles{ProfileCount}.rlProfileCfg_t.txOutPowerBackoffCode(3:end)), -16), 255);
        end
        
        % Chirp configuration
        Params.DevConfig(DevId).NumChirps = 0;
        for ChirpCount = 1:NumChirpBlocks
            Params.DevConfig(DevId).NumChirps = Params.DevConfig(DevId).NumChirps + (Chirps{ChirpCount}.rlChirpCfg_t.chirpEndIdx - Chirps{ChirpCount}.rlChirpCfg_t.chirpStartIdx) + 1;
            for ChirpId = Chirps{ChirpCount}.rlChirpCfg_t.chirpStartIdx+1:Chirps{ChirpCount}.rlChirpCfg_t.chirpEndIdx+1
                Params.DevConfig(DevId).Chirp(ChirpId).ChirpIdx 	 = ChirpId;
				Params.DevConfig(DevId).Chirp(ChirpId).ProfileId	 = Chirps{ChirpCount}.rlChirpCfg_t.profileId;
				Params.DevConfig(DevId).Chirp(ChirpId).StartFreqVar	 = Chirps{ChirpCount}.rlChirpCfg_t.startFreqVar_MHz;
				Params.DevConfig(DevId).Chirp(ChirpId).FreqSlopeVar	 = Chirps{ChirpCount}.rlChirpCfg_t.freqSlopeVar_KHz_usec;
				Params.DevConfig(DevId).Chirp(ChirpId).IdleTimeVar	 = Chirps{ChirpCount}.rlChirpCfg_t.idleTimeVar_usec;
				Params.DevConfig(DevId).Chirp(ChirpId).AdcStartTime	 = Chirps{ChirpCount}.rlChirpCfg_t.adcStartTimeVar_usec;
				Params.DevConfig(DevId).Chirp(ChirpId).Tx0Enable 	 = bitget(hex2dec(Chirps{ChirpCount}.rlChirpCfg_t.txEnable(3:end)),1);
				Params.DevConfig(DevId).Chirp(ChirpId).Tx1Enable 	 = bitget(hex2dec(Chirps{ChirpCount}.rlChirpCfg_t.txEnable(3:end)),2);
				Params.DevConfig(DevId).Chirp(ChirpId).Tx2Enable 	 = bitget(hex2dec(Chirps{ChirpCount}.rlChirpCfg_t.txEnable(3:end)),3);
            end
        end
        
        % BPM configuration for each chirp
        NumBpmBlocks = length(MmWaveDevicesConfig{Count}.rfConfig.rlBpmChirps);
		BpmConfig = MmWaveDevicesConfig{Count}.rfConfig.rlBpmChirps;
        for BpmCount = 1:NumBpmBlocks
			for ChirpId = BpmConfig{BpmCount}.rlBpmChirpCfg_t.chirpStartIdx+1:BpmConfig{BpmCount}.rlBpmChirpCfg_t.chirpEndIdx+1
				Params.DevConfig(DevId).Chirp(ChirpId).Tx0OffBpmVal	 = bitget(hex2dec(BpmConfig{BpmCount}.rlBpmChirpCfg_t.constBpmVal(3:end)), 1);
				Params.DevConfig(DevId).Chirp(ChirpId).Tx0OnBpmVal	 = bitget(hex2dec(BpmConfig{BpmCount}.rlBpmChirpCfg_t.constBpmVal(3:end)), 2);
				Params.DevConfig(DevId).Chirp(ChirpId).Tx1OffBpmVal	 = bitget(hex2dec(BpmConfig{BpmCount}.rlBpmChirpCfg_t.constBpmVal(3:end)), 3);
				Params.DevConfig(DevId).Chirp(ChirpId).Tx1OnBpmVal	 = bitget(hex2dec(BpmConfig{BpmCount}.rlBpmChirpCfg_t.constBpmVal(3:end)), 4);
				Params.DevConfig(DevId).Chirp(ChirpId).Tx2OffBpmVal	 = bitget(hex2dec(BpmConfig{BpmCount}.rlBpmChirpCfg_t.constBpmVal(3:end)), 5);
				Params.DevConfig(DevId).Chirp(ChirpId).Tx2OnBpmVal	 = bitget(hex2dec(BpmConfig{BpmCount}.rlBpmChirpCfg_t.constBpmVal(3:end)), 6);
			end
        end
		
        % Phase shifter configuration for each chirp
		NumPhaseShiftBlocks = length(MmWaveDevicesConfig{Count}.rfConfig.rlRfPhaseShiftCfgs);
		PhaseShiftConfig = MmWaveDevicesConfig{Count}.rfConfig.rlRfPhaseShiftCfgs;
        for PhaseShiftCount = 1:NumPhaseShiftBlocks
			for ChirpId = PhaseShiftConfig{PhaseShiftCount}.rlRfPhaseShiftCfg_t.chirpStartIdx+1:PhaseShiftConfig{PhaseShiftCount}.rlRfPhaseShiftCfg_t.chirpEndIdx+1
				Params.DevConfig(DevId).Chirp(ChirpId).Tx0PhaseShift = PhaseShiftConfig{PhaseShiftCount}.rlRfPhaseShiftCfg_t.tx0PhaseShift * 5.625;
				Params.DevConfig(DevId).Chirp(ChirpId).Tx1PhaseShift = PhaseShiftConfig{PhaseShiftCount}.rlRfPhaseShiftCfg_t.tx1PhaseShift * 5.625;
				Params.DevConfig(DevId).Chirp(ChirpId).Tx2PhaseShift = PhaseShiftConfig{PhaseShiftCount}.rlRfPhaseShiftCfg_t.tx2PhaseShift * 5.625;
            end
        end
        
        % Data Format
        AdcFormatConfig = MmWaveDevicesConfig{Count}.rfConfig.rlAdcOutCfg_t.fmt;
        Params.DevConfig(DevId).DataFormat.NumAdcBits       = AdcFormatConfig.b2AdcBits;
        Params.DevConfig(DevId).DataFormat.Format			= AdcFormatConfig.b2AdcOutFmt;
        Params.DevConfig(DevId).DataFormat.ReductionFactor  = AdcFormatConfig.b8FullScaleReducFctr;      
        Params.DevConfig(DevId).DataFormat.IQSwap			= MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevDataFmtCfg_t.iqSwapSel;
        
        % Data Path
        DataPathConfig = MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevDataPathCfg_t;
        Params.DevConfig(DevId).DataPath.Interface          = DataPathConfig.intfSel;
        Params.DevConfig(DevId).DataPath.Packet0            = hex2dec(DataPathConfig.transferFmtPkt0(3:end));
        Params.DevConfig(DevId).DataPath.Packet1            = hex2dec(DataPathConfig.transferFmtPkt1(3:end));
        Params.DevConfig(DevId).DataPath.CqConfig           = DataPathConfig.cqConfig;
        Params.DevConfig(DevId).DataPath.Cq0TransSize       = DataPathConfig.cq0TransSize;
        Params.DevConfig(DevId).DataPath.Cq1TransSize       = DataPathConfig.cq1TransSize;
        Params.DevConfig(DevId).DataPath.Cq2TransSize       = DataPathConfig.cq2TransSize;
        
        if (Params.DevConfig(DevId).DataPath.Interface == 1)
            Params.DevConfig(DevId).DataPath.LaneMap			= hex2dec(MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevLaneEnable_t.laneEn(3:end));
            Params.DevConfig(DevId).DataPath.LvdsFormat			= hex2dec(MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevLvdsLaneCfg_t.laneFmtMap(3:end));
            Params.DevConfig(DevId).DataPath.LvdsMsbFirst		= bitget(hex2dec(MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevLvdsLaneCfg_t.laneParamCfg(3:end)), 1);
            Params.DevConfig(DevId).DataPath.LvdsMsbCrcPresent	= bitget(hex2dec(MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevLvdsLaneCfg_t.laneParamCfg(3:end)), 2);
            Params.DevConfig(DevId).DataPath.LvdsPktEndPulse	= bitget(hex2dec(MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevLvdsLaneCfg_t.laneParamCfg(3:end)), 3);
        end
        
        if (Params.DevConfig(DevId).DataPath.Interface == 0)
            Params.DevConfig(DevId).DataPath.CsiLane0Pos        = bitand(bitshift(hex2dec(MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevCsi2Cfg_t.lanePosPolSel(3:end)), 0), 15);
            Params.DevConfig(DevId).DataPath.CsiLane1Pos        = bitand(bitshift(hex2dec(MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevCsi2Cfg_t.lanePosPolSel(3:end)), -4), 15);
            Params.DevConfig(DevId).DataPath.CsiLane2Pos        = bitand(bitshift(hex2dec(MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevCsi2Cfg_t.lanePosPolSel(3:end)), -8), 15);
            Params.DevConfig(DevId).DataPath.CsiLane3Pos        = bitand(bitshift(hex2dec(MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevCsi2Cfg_t.lanePosPolSel(3:end)), -12), 15);
            Params.DevConfig(DevId).DataPath.CsiLaneClkPos      = bitand(bitshift(hex2dec(MmWaveDevicesConfig{Count}.rawDataCaptureConfig.rlDevCsi2Cfg_t.lanePosPolSel(3:end)), -16), 15);
        end
       Params.FrameType = 0;
        if (Params.FrameType == 0)
            FrameConfig = MmWaveDevicesConfig{Count}.rfConfig.rlFrameCfg_t;   

            % Frame Configuration
            Params.DevConfig(DevId).FrameConfig.ChirpIdx            = FrameConfig.chirpStartIdx;
            Params.DevConfig(DevId).FrameConfig.ChirpEndIdx         = FrameConfig.chirpEndIdx;
            Params.DevConfig(DevId).FrameConfig.NumChirpLoops       = FrameConfig.numLoops;
            Params.DevConfig(DevId).FrameConfig.NumFrames           = FrameConfig.numFrames;
            Params.DevConfig(DevId).FrameConfig.Periodicity         = FrameConfig.framePeriodicity_msec;
            Params.DevConfig(DevId).FrameConfig.FrameTriggerDelay   = FrameConfig.frameTriggerDelay;
        end
        
        if (Params.FrameType == 1)
            % Advanced Frame Configuration
            AdvancedFrameSequence = MmWaveDevicesConfig{Count}.rfConfig.rlAdvFrameCfg_t.frameSeq;
            AdvancedFrameData = MmWaveDevicesConfig{Count}.rfConfig.rlAdvFrameCfg_t.frameData;
            Params.DevConfig(DevId).AdvFrame.NumFrames          = AdvancedFrameSequence.numFrames;					
            Params.DevConfig(DevId).AdvFrame.NumSubFrames		= AdvancedFrameSequence.numOfSubFrames;					
            Params.DevConfig(DevId).AdvFrame.FrameTriggerDelay	= AdvancedFrameSequence.frameTrigDelay_usec;

            % Sub-Frame Configuration
            for SubFrameId = 1:length(AdvancedFrameSequence.subFrameCfg)								
                Params.DevConfig(DevId).AdvFrame.SubFrame(SubFrameId).ForceProfileIdx       = AdvancedFrameSequence.subFrameCfg{SubFrameId}.rlSubFrameCfg_t.forceProfileIdx;
                Params.DevConfig(DevId).AdvFrame.SubFrame(SubFrameId).ChirpStartIdx         = AdvancedFrameSequence.subFrameCfg{SubFrameId}.rlSubFrameCfg_t.chirpStartIdx;
                Params.DevConfig(DevId).AdvFrame.SubFrame(SubFrameId).NumChirp              = AdvancedFrameSequence.subFrameCfg{SubFrameId}.rlSubFrameCfg_t.numOfChirps;
                Params.DevConfig(DevId).AdvFrame.SubFrame(SubFrameId).NumChirpLoops         = AdvancedFrameSequence.subFrameCfg{SubFrameId}.rlSubFrameCfg_t.numLoops;
                Params.DevConfig(DevId).AdvFrame.SubFrame(SubFrameId).BurstPeriod           = AdvancedFrameSequence.subFrameCfg{SubFrameId}.rlSubFrameCfg_t.burstPeriodicity_msec;
                Params.DevConfig(DevId).AdvFrame.SubFrame(SubFrameId).ChirpStartIdOffset    = AdvancedFrameSequence.subFrameCfg{SubFrameId}.rlSubFrameCfg_t.chirpStartIdxOffset;
                Params.DevConfig(DevId).AdvFrame.SubFrame(SubFrameId).NumBurst              = AdvancedFrameSequence.subFrameCfg{SubFrameId}.rlSubFrameCfg_t.numOfBurst;
                Params.DevConfig(DevId).AdvFrame.SubFrame(SubFrameId).NumBurstLoops         = AdvancedFrameSequence.subFrameCfg{SubFrameId}.rlSubFrameCfg_t.numOfBurstLoops;
                Params.DevConfig(DevId).AdvFrame.SubFrame(SubFrameId).SubFramePeriod        = AdvancedFrameSequence.subFrameCfg{SubFrameId}.rlSubFrameCfg_t.subFramePeriodicity_msec;
                Params.DevConfig(DevId).AdvFrame.SubFrame(SubFrameId).ChirpsPerDataPkt      = AdvancedFrameData.subframeDataCfg{SubFrameId}.rlSubFrameDataCfg_t.numChirpsInDataPacket;
            end
        end
        
        if (Params.FrameType == 2)
            % Continuous mode configuration
            ContinuousModeConfig = MmWaveDevicesConfig{Count}.rfConfig.rlContModeCfg_t;
            Params.DevConfig(DevId).ContFrame.StartFreq                 = ContinuousModeConfig.startFreqConst_GHz;
            Params.DevConfig(DevId).ContFrame.SamplingRate              = ContinuousModeConfig.digOutSampleRate;
            Params.DevConfig(DevId).ContFrame.Tx0OutPowerBackoffCode	= bitand(bitshift(hex2dec(ContinuousModeConfig.txOutPowerBackoffCode(3:end)), 0), 255);
            Params.DevConfig(DevId).ContFrame.Tx1OutPowerBackoffCode	= bitand(bitshift(hex2dec(ContinuousModeConfig.txOutPowerBackoffCode(3:end)), -8), 255);
            Params.DevConfig(DevId).ContFrame.Tx2OutPowerBackoffCode	= bitand(bitshift(hex2dec(ContinuousModeConfig.txOutPowerBackoffCode(3:end)), -16), 255);
            Params.DevConfig(DevId).ContFrame.Tx0PhaseShifter           = bitand(bitshift(hex2dec(ContinuousModeConfig.txPhaseShifter(3:end)), 0), 255);
            Params.DevConfig(DevId).ContFrame.Tx1PhaseShifter           = bitand(bitshift(hex2dec(ContinuousModeConfig.txPhaseShifter(3:end)), -8), 255);
            Params.DevConfig(DevId).ContFrame.Tx2PhaseShifter           = bitand(bitshift(hex2dec(ContinuousModeConfig.txPhaseShifter(3:end)), -16), 255);
            Params.DevConfig(DevId).ContFrame.RxGain                    = ContinuousModeConfig.rxGain_dB;
            Params.DevConfig(DevId).ContFrame.HpfCornerFreq1            = ContinuousModeConfig.hpfCornerFreq1;
            Params.DevConfig(DevId).ContFrame.HpfCornerFreq2            = ContinuousModeConfig.hpfCornerFreq2;
        end
    end
	Params.TxToEnable = TxToEnable;
	Params.RxToEnable = RxToEnable;	
end