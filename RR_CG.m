
function conFilter=RR_CG(RX,RN,r,ranks,sigma_square)
x_0=zeros(size(r));
[~,R_total,diag_values,alpha_set]=conjgrad(RX,r,x_0,ranks);

U=R_total(:,1:ranks);
cg_diag=diag_values(1:ranks);
alpha=alpha_set(1:ranks);
% alpha=(U'*r)./cg_diag;



inv_sqrt_cg=1./(sqrt(cg_diag));
U_scale=U.*inv_sqrt_cg';
M=U_scale'*RN*U_scale;
M=(M+M')/2;


[D,EIG_VALUE_sub_vector]=eig(M);

c=D'*(sqrt(cg_diag).*alpha);


x0=0;
f=@(x) sum(EIG_VALUE_sub_vector.*c.^2./(1+x*EIG_VALUE_sub_vector).^2)-sigma_square;
g=@(x)-2*sum(EIG_VALUE_sub_vector.^2.*c.^2./(1+x*EIG_VALUE_sub_vector).^3);
mu=newton_root_power(f,g,x0);

y=D*(1./(1+mu*EIG_VALUE_sub_vector).*c);
x=y.*inv_sqrt_cg;
conFilter=U*x;
end



function [x,R_total,diag_values,alpha_set,beta_set] = conjgrad(A, b, x,iteration_num)
N=length(b);
R_total=zeros(N,iteration_num);
r = b - A * x;
p = r;
rsold = r' * r;
R_total(:,1)=p;


diag_values=zeros(iteration_num,1);
alpha_set=zeros(iteration_num,1);
beta_set=zeros(iteration_num,1);

for i = 1:iteration_num
    Ap = A * p;
    diag_values(i,1)=p'*Ap;


    alpha = rsold / diag_values(i,1);
    x = x + alpha * p;
    r = r - alpha * Ap;
    rsnew = r' * r;
%     if sqrt(rsnew) < 1e-10
%         break;
%     end
    beta_set(i)=rsnew / rsold ;

    p = r + beta_set(i) * p;
    rsold = rsnew;
    R_total(:,i+1)=p;


    alpha_set(i,1)=alpha;
end

end

function x0=newton_root_power(f,fprime,x0,maxIterations,tolerance)
if nargin<=3
    maxIterations=1e4;
    tolerance=1e-7;
end

for i = 1 : maxIterations

 y = f(x0);
if y<=0
    break;
end


 yprime = fprime(x0);

 if(abs(yprime) < eps) %Don't want to divide by too small of a number
 % denominator is too small
 break; %Leave the loop
 end

 x1 = x0 - y/yprime; %Do Newton's computation

 if(norm(x1 - x0)/norm(x0) <= tolerance) %If the result is within the desired tolerance
    break; %Done, so leave the loop
 end

 x0 = x1; %Update x0 to start the process again

end
end
