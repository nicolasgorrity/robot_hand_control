function relevance_fourier_descriptors

close all

N=200;
Nangles=100;% Number of generated rotated images
dico=6;    % Number of clusters
numiter=400;
cmax=10;    % Number of Fourier coefficients
cmin=-cmax;

list_angles=180*(1-2*rand(Nangles,1));
list_=dir('../database/*.bmp');
Nimages=length(list_);

% Display binary images
for n=1:Nimages
    filename = strcat(list_(n).folder,'\',list_(n).name);
    ima=imread(filename);
    ima=double(ima>0.5);
    figure(2);
    subplot(3,5,n)
    imshow(ima)
    drawnow
end

% Coefficients k \in [cmin, cmax] pour chaque image g�n�r�e
tabcoeff=zeros(cmax-cmin+1,length(list_angles));    % zeros(n,m) = matrice rectangulaire hauteur n largeur m
% Real and imaginary part of Fourier coefficients
vectors=zeros(2*(cmax-cmin+1),length(list_angles));
% Reconstructed edges which size are N+1
tabcontfil=zeros(N+1,length(list_angles));

for n=1:length(list_angles)
    choix=randi(Nimages,1);                         % Choose image randomly
    filename = strcat(list_(choix).folder,'\',list_(choix).name);
    ima=double(imread(filename)>0.5);               % Get binary representation of image
    imar=imrotate(ima,list_angles(n),'crop');       % Rotate image according to random angle
    z=get_contour_pixels(imar);                              % Get edges of rotated image -> z contains all the edge points encoded by z=x+iy
    [coeff,ncoeff]=fourier_descriptors(z,cmax);  % Computes Fourier coefficients of the edge
    tabcoeff(:,n)=coeff;                            % Store Fourier coefficients of this image
    contfil=contour_reconstruction(coeff,N,cmax);   % Reconstruct edge from Fourier coefficients
    tabcontfil(:,n)=contfil;                        % Store reconstructed edges
    vectors(:,n)=[real(coeff);imag(coeff)];        % Store real part an imaginary part of Fourier coefficients
    n
end
tabcoeff
%%
% Display all reconstructed edges
figure
plot(real(tabcontfil),imag(tabcontfil),'-',real(tabcontfil(1,:)),imag(tabcontfil(1,:)),'o')
title('contours associ�es aux coefficients normalis�s')
grid on
axis equal
axis ij
drawnow

% Display coefficients modules
figure
plot(ncoeff,abs(tabcoeff))
title('module des coefficients')
xlabel('k');
ylabel('d_k');
grid on
zoom on
drawnow

% Display real and imaginary part of coefficients
figure
plot(vectors)
title('partie r�elle et imaginaire des coefficients normalis�s')
set(gca,'XTick',1:size(vectors,1))
set(gca,'XTickLabel',[ncoeff ncoeff])
grid on
ech=axis;
axis([1 size(vectors,1) ech(3:4)])
drawnow


[vectors_list,code,occur]=kmeans(vectors,dico,numiter);

figure
for n=1:dico
    contfil=resconstrdesfour(vectors_list(1:end/2,n)+1i*vectors_list(end/2+1:end,n),N,cmax);
    subplot(4,4,n)
    h=plot(real(contfil),imag(contfil),'-',real(contfil(1)),imag(contfil(1)),'o');
    title(['prototype ' int2str(n)])
    set(h(1),'LineWidth',2)
    set(h(2),'LineWidth',3)
    grid on
    axis equal
    axis ij
    for i=(1:length(list_angles))
       if (code(i)==n-1)
          hold on;
          plot(real(tabcontfil(:,i)),imag(tabcontfil(:,i)),'Color',[rand, rand, rand]);
       end
    end
    hold on
    drawnow
end

%%
colors = [[1 0 0]; [0 1 0]; [0 0 1]; [1 1 0]; [0 1 1]; [1 0 1]];

figure
for i=cmin:cmax
    subplot(5,5,i-cmin+1);
    scatter(real(tabcoeff(i-cmin+1,:)), imag(tabcoeff(i-cmin+1,:)), 36, colors(code+1,:));
    title(strcat('k=',int2str(i)));
    grid on
    drawnow
end
figure
for i=1:500
    hold on
    scatter(ncoeff, abs(tabcoeff(:,i))', 36, colors(code(i)+1,:));
    grid on
    drawnow
end
