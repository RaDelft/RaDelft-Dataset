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

function [fileNameStruct]= getBinFileNames_withIdx(folderName, fileIdx)
% Description : Given a folder name and fileIdx it returns the FileName corresponding
% to the Master, slave1, slave2, slave3 binary files
% Input : folderName : The name of the folder in which the data is saved
% Output : 

currentFolder = pwd;
cd(folderName);
listing = dir(strcat('*',fileIdx,'_data.bin'));
listing_idx = dir(strcat('*',fileIdx,'_idx.bin'));

cd(currentFolder);

if (length(listing)>4)   
    error('Too many binary Files corresponding to the same fileIdx')
else
    
for ii=1:length(listing)    
    if (strfind(listing(ii).name,'master'))
        fileNameStruct.master = listing(ii).name;
    end
    if (strfind(listing(ii).name,'slave1'))
        fileNameStruct.slave1 = listing(ii).name;
    end
    if (strfind(listing(ii).name,'slave2'))
        fileNameStruct.slave2 = listing(ii).name;
    end
    if (strfind(listing(ii).name,'slave3'))
        fileNameStruct.slave3 = listing(ii).name;
    end
    if (strfind(listing_idx(ii).name,'master'))
        fileNameStruct.masterIdxFile = listing_idx(ii).name;
    end
    if (strfind(listing_idx(ii).name,'slave1'))
        fileNameStruct.slave1IdxFile = listing_idx(ii).name;
    end
    if (strfind(listing_idx(ii).name,'slave2'))
        fileNameStruct.slave2IdxFile = listing_idx(ii).name;
    end
    if (strfind(listing_idx(ii).name,'slave3'))
        fileNameStruct.slave3IdxFile = listing_idx(ii).name;
    end
end
    fileNameStruct.dataFolderName = folderName;
    
end