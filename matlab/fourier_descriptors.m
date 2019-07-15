function [coeff,num]=fourier_descriptors(z,cmax)
cmin=-cmax;
z_moy=mean(z);
longc=length(z);
% Fourier coefficients
TC=fft(z-z_moy)/longc;
num=cmin:cmax;

% select coeff between cmin et cmax
coeff=zeros(cmax-cmin+1,1);
coeff(end-cmax:end)=TC(1:cmax+1);
coeff(1:-cmin)=TC(end+cmin+1:end);

% reverse sequence if contour is browsed in non-trigonometric direction
if abs(coeff(num==-1))>abs(coeff(num==1))
    coeff=coeff(end:-1:1);
end

% phasis correction for rotation- and start-point- invariance
Phi=angle(coeff(num==-1).*coeff(num==1))/2;
coeff=coeff*exp(-1i*Phi);

theta=angle(coeff(num==1));
coeff=coeff.*exp(-1i*num'*theta);

% scale-invariance
coeff=coeff/abs(coeff(num==1));
