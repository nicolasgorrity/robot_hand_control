close all

N=200;
Nangles=1000;% Number of generated rotated images
cmax=10;    % Number of Fourier coefficients
cmin=-cmax;

%% Display input images
disp('Displaying dataset of images');

list_angles=180*(1-2*rand(Nangles,1));
list_=dir('database/*.bmp');
Nimages=length(list_);

% Display binary images
for n=1:15
    filename = strcat(list_(n).folder,'\',list_(n).name);
    ima=imread(filename);
    ima=double(ima>0.5);
    figure(2);
    subplot(3,5,n)
    imshow(ima)
    drawnow
end

%% Get Y labels
disp('Getting images labels');

labels_files=dir('database/*.txt');
labels = zeros(Nimages,1);

% Display binary images
for n=1:Nimages
    filename = strcat(labels_files(n).folder,'\',labels_files(n).name);
    labels(n)=textread(filename);
end

%% Get X input parameters and Y labels
disp('Computing Fourier descriptor of training set');

% Coefficients k \in [cmin, cmax] pour chaque image g�n�r�e
tabcoeff=zeros(2*(cmax-cmin+1),length(list_angles));
% Real labels
real_labels = zeros(1, length(list_angles));
% Count of each class
class_count = zeros(1, 4);
n=0;

while ((class_count(1)<200) || (class_count(2)<200) || (class_count(3)<200) || (class_count(4)<200))
    n=mod(n+1, Nangles);
    choice=randi(Nimages,1);            % Choose image randomly
    filename = strcat(list_(choice).folder,'\',list_(choice).name);
    ima=double(imread(filename)>0.5);               % Get binary representation of image
    flip_o_not = randi(2,1);
    if (flip_o_not==1)
        ima=flip(ima,2);
    end
    imar=imrotate(ima,list_angles(n),'crop');       % Rotate image according to random angle
    figure(2);
    imshow(imar);
    z=get_contour_pixels(imar);                              % Get edges of rotated image -> z contains all the edge points encoded by z=x+iy
    [coefficients,~]=fourier_descriptors(z,cmax);       % Computes Fourier coefficients of the edge
    tabcoeff(:,n)=[real(coefficients);imag(coefficients)];        % Store Fourier coefficients of this image
    real_labels(n) = labels(choice);
    class_count(labels(choice)) = class_count(labels(choice)) + 1;
    n
end

X = tabcoeff';
Y = real_labels;

%% PCA
[coeff,score,latent,tsquared,explained,mu] = pca(X);
% Find number of principal components to reach 99% of variance
Ncomp = 0;
variance = 0;
for i=1:length(explained)
    Ncomp = Ncomp + 1;
    variance = variance + explained(i);
    if (variance >= 99)
        break;
    end
end
disp([num2str(Ncomp), ' principal components represent ', num2str(variance), ' % of global variance.']);
X_train_PCA = score(:, 1:Ncomp);

%% Save CSV files
dlmwrite('dataset_gen/pca_coeff.csv', coeff, 'delimiter', ',', 'precision', 9);
dlmwrite('dataset_gen/pca_inv_coeffT.csv', inv(coeff'), 'delimiter', ',', 'precision', 9);
dlmwrite('dataset_gen/pca_mu.csv', mu, 'delimiter', ',', 'precision', 9);
dlmwrite('dataset_gen/pca_Xtrain.csv', X_train_PCA, 'delimiter', ',', 'precision', 9);
dlmwrite('dataset_gen/pca_Ytrain.csv', Y, 'delimiter', ',', 'precision', 9);

disp("Dataset saved.");
