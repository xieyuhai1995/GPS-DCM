function [Times, MinESS, MedianESS, MaxESS, MeanESS, MinACT, MedianACT, MaxACT, MeanACT] = CalculateStatistics( Method, DataSet )

% Get stats
Files = dir(['*' Method '*' DataSet '*.mat']);


for i = 1:length(Files)

    Data     = open(Files(i).name);
    
    ESS{i}   = CalculateESS(Data.betaPosterior, length(Data.betaPosterior)-1);
    
    MinESS(i)    = min(ESS{i});
    MaxESS(i)    = max(ESS{i});
    MedianESS(i) = median(ESS{i});
    MeanESS(i)   = mean(ESS{i});
    Times(i)     = Data.TimeTaken;
    
    
    ACT{i}   = CalculateACT(Data.betaPosterior);
    MinACT(i)    = min(ACT{i});
    MaxACT(i)    = max(ACT{i});
    MedianACT(i) = median(ACT{i});
    MeanACT(i)   = mean(ACT{i});
    
end

% disp(['Time:   ' num2str(mean(Times)) ' +/- ' num2str(std(Times)/sqrt(length(Times)))])
% disp(' ')

disp(['Time per iteration:   ' num2str(mean(Times)/size(Data.betaPosterior,1))])
disp(' ')

disp(['ESS for ' Method ' with ' DataSet ' dataset.'])
disp(' ')

disp(['Min:    ' num2str(mean(MinESS)) ' +/- ' num2str(std(MinESS)/sqrt(length(MinESS)))])
disp(['Median: ' num2str(mean(MedianESS)) ' +/- ' num2str(std(MedianESS)/sqrt(length(MedianESS)))])
disp(['Mean:   ' num2str(mean(MeanESS)) ' +/- ' num2str(std(MeanESS)/sqrt(length(MeanESS)))])
disp(['Max:    ' num2str(mean(MaxESS)) ' +/- ' num2str(std(MaxESS)/sqrt(length(MaxESS)))])

disp('')
disp(['Min ESS per second: ' num2str(mean(MinESS)/mean(Times))])
disp(' ')


disp(['ACT for ' Method ' with ' DataSet ' dataset.'])
disp(' ')

disp(['Min:    ' num2str(mean(MinACT)) ' +/- ' num2str(std(MinACT)/sqrt(length(MinACT)))])
disp(['Median: ' num2str(mean(MedianACT)) ' +/- ' num2str(std(MedianACT)/sqrt(length(MedianACT)))])
disp(['Mean:   ' num2str(mean(MeanACT)) ' +/- ' num2str(std(MeanACT)/sqrt(length(MeanACT)))])
disp(['Max:    ' num2str(mean(MaxACT)) ' +/- ' num2str(std(MaxACT)/sqrt(length(MaxACT)))])

disp('')
disp(['Time multiply Mean ACT: ' num2str(mean(Times)*mean(MeanACT))])

end
