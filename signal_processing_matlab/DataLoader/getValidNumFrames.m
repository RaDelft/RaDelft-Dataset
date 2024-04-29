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

% get number of valid frames in the associated *_data.bin file captured
% with TDA2 platform

% File header in *_idx.bin:
%     struct Info
%     {
%         uint32_t tag;
%         uint32_t version;
%         uint32_t flags;
%         uint32_t numIdx;       // number of frames 
%         uint64_t dataFileSize; // total data size written into file
%     };
% 
% Index for every frame from each radar:
%     struct BuffIdx
%     {
%         uint16_t tag;
%         uint16_t version; /*same as Info.version*/
%         uint32_t flags;
%         uint16_t width;
%         uint16_t height;
%         uint32_t pitchOrMetaSize[4]; /*For image data, this is pitch.
%                                                        For raw data, this is size in bytes per metadata plane.*/
%         uint32_t size; /*total size in bytes of the data in the buffer (sum of all planes)*/
%         uint64_t timestamp;
%         uint64_t offset;
%     };


function [numIdx dataFileSize] = getValidNumFrames(adcIdxFileName)

idxFile = fopen(adcIdxFileName,'r');
heaferInfoSize = 6;
heaferInfo = fread(idxFile, heaferInfoSize,'uint32');
numIdx = heaferInfo(4); % number of effective frame
fclose(idxFile);
idxFile = fopen(adcIdxFileName,'r');
heaferInfoSize = 3;
heaferInfo = fread(idxFile, heaferInfoSize,'uint64');
dataFileSize = heaferInfo(3); % data size for the effective number of frames
fclose(idxFile);

end