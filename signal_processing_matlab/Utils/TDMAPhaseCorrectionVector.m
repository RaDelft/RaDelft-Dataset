function [dopplerCorrectedCube, dopplerIndex] = TDMAPhaseCorrectionVector(preAngleFFT)

overlapAntenna_ID = load('overlapAntenna_ID_1TX.mat').overlapAntenna_ID_1TX;

rangeSize = size(preAngleFFT,1);
dopplerSize = size(preAngleFFT,2);

TDM_MIMO_numTX = 12;
numRxAnt = 16;

dopplerCorrectedCube = zeros(size(preAngleFFT));
dopplerIndex = zeros(rangeSize, dopplerSize, 'int16');

parfor rangeBin=1:rangeSize
    %fprintf('RANGE BIN: %d \n', rangeBin)
    dopplerBins = 1:dopplerSize;
    dopplerInd_unwrap = zeros(dopplerSize, TDM_MIMO_numTX);
    dopplerInd_unwrap(dopplerBins > dopplerSize/2,:) = dopplerBins(dopplerBins> dopplerSize/2)' + ((1:TDM_MIMO_numTX)-(TDM_MIMO_numTX/2+1))*dopplerSize;
    dopplerInd_unwrap(dopplerBins <= dopplerSize/2,:) = dopplerBins(dopplerBins<= dopplerSize/2)' + ((1:TDM_MIMO_numTX)-TDM_MIMO_numTX/2)*dopplerSize;

    sig_bin_org = squeeze(preAngleFFT(rangeBin, dopplerBins,:));

    %Doppler phase correction due to TDM MIMO
    deltaPhi = 2*pi*(dopplerInd_unwrap-dopplerSize/2)/(TDM_MIMO_numTX*dopplerSize);

    % construct all possible signal vectors based on the number
    % of possible hypothesis
    sig_bin = zeros(dopplerSize, 192, TDM_MIMO_numTX);
    for i_TX = 1:TDM_MIMO_numTX
        RX_ID = (i_TX-1)*numRxAnt+1 : i_TX*numRxAnt;
        A = sig_bin_org(:,RX_ID);
        A = reshape(A,[dopplerSize,16,1]);
        B = reshape(exp(-1j*(i_TX-1)*deltaPhi), [dopplerSize,1,12]);
        sig_bin(:,RX_ID,: )= A.*B;
    end
    
    % use overlap antenna to do max velocity unwrap
    signal_overlap = sig_bin_org(:,overlapAntenna_ID(:,1:2));
    signal_overlap = reshape(signal_overlap,dopplerSize,32,2);

    %check the phase difference of each overlap antenna pair
    %for each hypothesis
    angle_sum_test = [];
    for i_sig = 1:size(signal_overlap,2)
        for i_test = 1:size(deltaPhi,2)
            % A = reshape(signal_overlap(:,1:i_sig,2), [], 1);
            % B = reshape(deltaPhi(:,i_test), 1, []);
            % signal2 = A.*exp(-1j*B);
            signal2 = signal_overlap(:,1:i_sig,2).*exp(-1j*deltaPhi(:,i_test));
            angle_sum_test(:,i_sig,i_test) = angle(sum(signal_overlap(:,1:i_sig,1).*conj(signal2),2));

        end
    end

    %chosee the hypothesis with minimum phase difference to
    %estimate the unwrap factor
    [val_doppler_unwrap_integ_overlap, doppler_unwrap_integ_overlap] = min(abs(angle_sum_test),[],3);

    for dopplerBin=1:dopplerSize
        a = doppler_unwrap_integ_overlap(dopplerBin,:);
        b = unique(a);
        c = histc(a(:),b);
        [val, ind] = max(c);
        doppler_unwrap_integ = b(ind);
        dopplerIndex(rangeBin, dopplerBin) = int16(dopplerInd_unwrap(dopplerBin, (doppler_unwrap_integ)));
        dopplerCorrectedCube(rangeBin, dopplerBin, :) = squeeze(sig_bin(dopplerBin,:,b(ind)));

    end


end

