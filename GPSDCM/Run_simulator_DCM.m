function [ObjFct, Z, new_data] = Run_simulator_DCM(x, trueData, IS, Ep, M, U, V, row, column)

% total number of time points = no of species * no of time points per species
ntp = size(trueData,1);

% extra_p contain the initial values for the species

n = size(x,1);

NLL = NaN(n,1); Z = NaN(ntp,1);
%text = 'Sample count ';
size_col=size(column,1);
for i = 1:n
    
    for j=1:size_col
        Ep.A(row(j),column(j)) = x(i,j);
    end
    
    r=size(Ep.A,1);
    Ep.decay=sparse(x(i,size_col+1));
    Ep.transit=sparse(x(i,size_col+1+1:size_col+1+r))';
    Ep.epsilon=sparse(x(i,size_col+1+r+1));
    Ep.a=sparse(x(i,size_col+1+r+1+1:size_col+1+r+1+2))';
    Ep.b=sparse(x(i,size_col+1+r+1+3:size_col+1+r+1+4))';
    Ep.c=sparse(x(i,size_col+1+r+1+4+1:size_col+1+r+1+4+r));

    [~,new_data] = spm_diff(IS,Ep,M,U,1,{V});
    new_data = spm_vec(new_data);

    Z = trueData - new_data; % for all 2 species

    NLL(i,:) = sum(abs(Z.^2)); % for every species

%    disp(strcat(text,num2str(i)));
    
end

ObjFct = NLL;

