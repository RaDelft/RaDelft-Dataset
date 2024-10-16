function [rangeBins,dopplerBins,azimuthBins] = detOSOSOS2(radarCube)

rGuardCell = 0;
rTrainCell = 10;
aGuardCell = 0;
aTrainCell = 10;
k = floor((rTrainCell*aTrainCell*4) * 0.75);
p = 1e-2;

rangeAngle = squeeze(sum(radarCube,2).^2);
rangeAnglePad = padarray(rangeAngle,[rGuardCell+rTrainCell aGuardCell+aTrainCell],'symmetric','both');

detector = phased.CFARDetector2D('TrainingBandSize',[rTrainCell,aTrainCell], ...
    'Rank',k,'GuardBandSize',[rGuardCell,aGuardCell], ...
    'ProbabilityFalseAlarm',p,'Method','OS', 'OutputFormat','Detection index');

cutmask = zeros(size(rangeAnglePad));
cutmask(rGuardCell+rTrainCell + 1 : size(rangeAnglePad,1) - (rGuardCell+rTrainCell), aGuardCell+aTrainCell+1:size(rangeAnglePad,2)-(aGuardCell+aTrainCell)) = 1;

[cutrow,cutcol] = find(cutmask);
cut = [cutrow,cutcol]' ;
detections = detector(rangeAnglePad, cut);
detections(1,:) = detections(1,:) - (rGuardCell+rTrainCell);
detections(2,:) = detections(2,:) - (aGuardCell+aTrainCell);

rangeBins = detections(1,:);
azimuthBins = detections(2,:);

rangeBinsAux = [];
azimuthBinsAux = [];
dopplerBinsAux = [];

for i=1:length(rangeBins)
    
    dopplerProfile = squeeze(radarCube(rangeBins(i),:,azimuthBins(i))).^2;
    OSdetector = phased.CFARDetector('NumTrainingCells',16,'NumGuardCells',0,'Method', 'OS', 'Rank', 12, 'ProbabilityFalseAlarm',p,'OutputFormat','Detection index');
    cutidx = 1:length(dopplerProfile);

    dopplerBin = OSdetector(dopplerProfile',cutidx);
    %[~,dopplerBin] = findpeaks(dopplerProfile, 'NPeaks', 1 );

    
    rangeBinsAux = [rangeBinsAux, repmat(rangeBins(i), 1, length(dopplerBin))];
    azimuthBinsAux = [azimuthBinsAux, repmat(azimuthBins(i), 1, length(dopplerBin))];
    dopplerBinsAux = [dopplerBinsAux,dopplerBin];

end

rangeBins = rangeBinsAux;
dopplerBins = dopplerBinsAux;
azimuthBins = azimuthBinsAux;

end
