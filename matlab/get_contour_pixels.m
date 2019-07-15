function Z=get_contour_pixels(img)
% Global variables
global indl
global indc
global z
global ima
% table of indices in row and col for browsing neighborhood
indl=[-1 -1 -1 0 1 1  1  0 -1];
indc=[-1  0  1 1 1 0 -1 -1 -1];
% list of points of contour
z=zeros(5000,1);
ima=img;
[N,M]=size(ima);

% find start point of contour
found=0;
for n=1:N
   for m=1:M
      if (ima(n,m)==1)&&(found==0)
         longc=get_contour(n,m);
         found=1;
      end
   end
end
% truncate list to keep only contour pixels
z(longc+1:end)=[];
Z=z;

function longc=get_contour(n,m)
% global variables
global indl
global indc
global z
global ima
% start point
z(1)=m+1i*n;                                % z = m + i.n
% value 2 marks starting point
ima(n,m)=2;
index=2;
longc=2;
end=0;
while end==0
   found=0;
   % browse neighborhood
   for k=1:8
      if (ima(n+indl(k),m+indc(k))==0)&&...
            (ima(n+indl(k+1),m+indc(k+1))==1)&&(found==0)
         found=1;
         % mark contour point
         ima(n+indl(k+1),m+indc(k+1))=2;
         % momorize coordinates
         z(index)=(m+indc(k+1))+1i*(n+indl(k+1));
         % update current point
         n=n+indl(k+1);
         m=m+indc(k+1);
         if(longc<index)
            longc=index;
         end
         index=index+1;
      end
   end
   if(found==0)
      % if no new point found, go backwards
      index=index-1;
      m=real(z(index));
      n=imag(z(index));
   end
   % when no more point has been found the contour search is done
   if index==1
      end=1;
   end
end
