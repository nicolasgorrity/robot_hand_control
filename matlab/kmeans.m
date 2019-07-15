% kmeans.m
% [vectorslist,code,occur]=kmeans(vectors,dico,numiter)
% vectors : vectors of dataset (column-wise)
% dico : number of classes
% numiter : number of iterations
% vectorslist : contains prototypes vectors
% code : contains classes indexes
% occur : number of elements in each class

function [vectorslist,code,occur]=kmeans(vectors,dico,numiter)

dimvec=size(vectors,1);
M=size(vectors,2);

fid=fopen('vectors','w');
fwrite(fid,M,'int');
fwrite(fid,dimvec,'int');
fwrite(fid,dico,'int');
fwrite(fid,numiter,'int');
fwrite(fid,vectors,'float');
fclose(fid);

% Launch binary
dos('quantvec vectors dict code');

% Read result files
fid=fopen('dict','r');
dimvec=fread(fid,1,'int');
dico=fread(fid,1,'int');
vectorsliste=fread(fid,[dimvec dico],'float');
occur=fread(fid,dico,'int');
fclose(fid);
disp('vectors in list')
disp([dimvec dico])

fid=fopen('code','r');
code=fread(fid,M,'int');
fclose(fid);
