function [ dataCubeW, win] = windowing( dataCube, winDim,winType)
% Generación de los coeficientes de la ventana
% Window type: 0 - rectangular, (no window)
%              1 - blackman
%              2 - Chebyshev
%              3 - Flat top weighted window
%              4 - Gaussian
%              5 - Hamming
%              6 - Hanning
%              7 - Kaiser
%              8 - Taylor
%              9 - Tukey, tapered cosine
if winType == 0
    dataCubeW=dataCube;
else
    switch winType
        case 1 
            win=blackman(size(dataCube,winDim))';
        case 2 
            win=chebwin(size(dataCube,winDim),50)';
        case 3 
            win=flattopwin(size(dataCube,winDim))';
        case 4 
            win=gausswin(size(dataCube,winDim))';
        case 5 
            win=hamming(size(dataCube,winDim))';
        case 6 
            win=hann(size(dataCube,winDim))';
        case 7 
            win=kaiser(size(dataCube,winDim), 10)';
        case 8 
            win=taylorwin(size(dataCube,winDim))';
        case 9 
            r = 0.25;% 0(rectangular) to 1 (hann)
                     %  Returns an L-point Tukey window in the column vector, w. A Tukey window is a rectangular window with the 
                     %  first and last r/2 percent of the samples equal to parts of a cosine. See Definitions for the equation 
                     %  that defines the Tukey window. r is a real number between 0 and 1. If you input r ? 0, you obtain a rectwin 
                     %  window. If you input r ? 1, you obtain a hann window. r defaults to 0.5.
            win=tukeywin(size(dataCube,winDim),r)';
    end         


% Permutación para adaptar ventana a la dimensión a enventanar
permuteDim =1:3; aux=permuteDim(winDim);permuteDim(winDim)=2; permuteDim(2)=aux;
WINDOW=permute(win,permuteDim);
% Replicación de lo coeficientes de la ventana para aplicarlos a todos los
% puntos del cubo
repetitionDim = [size(dataCube,1) size(dataCube,2) size(dataCube,3)]; repetitionDim(winDim)=1;
WINDOW=repmat(WINDOW,repetitionDim(1),repetitionDim(2),repetitionDim(3));

% Aplicación de la ventana al cubo
dataCubeW=WINDOW.*dataCube;

end
end