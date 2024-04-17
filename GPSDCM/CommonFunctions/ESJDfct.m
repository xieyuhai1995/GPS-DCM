function ESJD = ESJDfct(x)
% Calculates the squared jumping distance between two points
% (mean squared distance between successive samples in x)

ESJD = mean( sum( ( x(2:end,:)-x(1:end-1,:) ).^2, 2 ) );

% nSamples = size(x,1);
% 
% sjd = 0; % squared jumping distance for the sequence of samples in x 
% %Calculate the mean squared distance between successive samples in x
% for i = 2:nSamples
%     sjd = sjd + (norm(x(i,:)-x(i-1,:))).^2;
% end
% ESJD = sjd/(nSamples-1);

end

