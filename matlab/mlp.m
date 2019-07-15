close all
Nclusters = 3;
cmax = 10;
cmin = -cmax;
Ntest = 1000;

%% Get dataset
list_angles=180*(1-2*rand(Ntest,1));
list_=dir('database/*.bmp');
Nimages=length(list_);

labels_files=dir('database/*.txt');
labels = zeros(Nimages,1);
for n=1:Nimages
    filename = strcat(labels_files(n).folder,'\',labels_files(n).name);
    labels(n)=textread(filename);
end

%% Get dataset
coeff = csvread('pca_coeff.csv');
mu = csvread('pca_mu.csv');
X_train_PCA = csvread('pca_Xtrain.csv');
Y = csvread('pca_Ytrain.csv');

%% Neural network training
disp("Training Neural Network");
% One hot encoding
Y_train = zeros(Nclusters, size(X_train_PCA,1));
for i=1:length(Y)
    Y_train(Y(i),i)=1;
end
net = newff(X_train_PCA', Y_train, 10, {'tansig' 'tansig'}, 'trainscg');
net.trainParam.epochs = 1000;
net.trainParam.lr = 0.02;
net.trainParam.show = 100;
net.trainParam.goal = 1e-10;
net.trainParam.min_grad = 1e-10;
net.divideParam.trainRatio = 3/5 * size(X_train_PCA,1);
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 2/5 * size(X_train_PCA,1);
net = train(net, X_train_PCA', Y_train);

%% Construct test set
disp('Computing Fourier descriptors of test set');
tabcoeff_test=zeros(2*(cmax-cmin+1),Ntest);
real_labels_test = zeros(1, Ntest);

for n=1:Ntest
    choice=randi(Nimages,1);                         % Choose image randomly
    filename = strcat(list_(choice).folder,'\',list_(choice).name);
    ima=double(imread(filename)>0.5);               % Get binary representation of image
    flip_o_not = randi(2,1);
    if (flip_o_not==1)
        ima=flip(ima,2);
    end
    imar=imrotate(ima,list_angles(n),'crop');       % Rotate image according to random angle
    z=get_contour_pixels(imar);                              % Get edges of rotated image -> z contains all the edge points encoded by z=x+iy
    [coefficients,~]=fourier_descriptors(z,cmax);  % Computes Fourier coefficients of the edge
    tabcoeff_test(:,n)=[real(coefficients);imag(coefficients)];    % Store Fourier coefficients of this image
    real_labels_test(n) = labels(choice);
    n
end

X_test = tabcoeff_test';
scores_test = (X_test-mu) * inv(coeff');
X_test_PCA = scores_test(:, 1:size(X_train_PCA, 2));

%% Predict test set
disp('MLP predicting test set');
% One hot encoding
Y_test = zeros(Nclusters, size(X_test_PCA,1));
for i=1:length(real_labels_test)
    Y_test(real_labels_test(i),i)=1;
end

[Y,Xf,Af] = sim(net, X_test_PCA');

Y_final = zeros(1, Ntest);
for i=1:Ntest
    [~, idx] = max(Y(:,i));
    Y_final(i) = idx;
end

disp('Empirical risk = ');
disp(sum(Y_final ~= real_labels_test) / Ntest);
