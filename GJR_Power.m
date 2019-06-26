% initializing GARCH parameters
alpha0 = 0.1;
alpha1=0.058;
gamma=0.068;
beta=0.826;

% N sample paths with n observations each
N=500;
n=1000;
% initializing PV, CI vectors and theta0
h = zeros(n,4);
p = zeros(n,4);
zval = zeros(n,4);
ci_alpha0 = zeros(2,n);
ci_beta = zeros(2,n);
ci_gamma = zeros(2,n);
ci_alpha1 = zeros(2,n);
theta0=[alpha0,beta,gamma,alpha1];


for i=1:n
    Mdl = gjr('Constant',alpha0,'ARCH',beta,'GARCH',alpha1,'leverage',gamma);
    [var,Y] = simulate(Mdl,N);
    % Setting up fmincon
    A = [0 1 1 1]; 
    b = 1;
    Aeq = [];
    Ceq = [];
    lb = [0 0 0 0];
    ub = [];
    options = optimoptions('fmincon','Display','off');
    [theta,fval,exitflag,output,lambda,grid,hessian] = fmincon(@(theta)loglikelihood(theta,Y),theta0, A, b, Aeq, Ceq, lb, ub,[],options);
    
    % calculate PV and CI
    sigma_hat=sqrt(abs(inv(hessian/N))/N);
    [h(i,1),p(i,1),ci_alpha0,zval(i,1)] = ztest(theta(1),alpha0,sigma_hat(1,1));
    [h(i,2),p(i,2),ci_beta,zval(i,2)] = ztest(theta(2),0,sigma_hat(2,2));
    [h(i,3),p(i,3),ci_gamma,zval(i,3)] = ztest(theta(3),gamma,sigma_hat(3,3));
    [h(i,4),p(i,4),ci_alpha1,zval(i,4)] = ztest(theta(4),alpha1,sigma_hat(4,4));
end 

count = zeros(1,4);
for i = 1:n
    for j = 1:4
        if h(i,j) == 1
            count(1,j) = count(1,j)+1;
        end
    end
end
rej_percent = count(1,:)./n;


function f = loglikelihood(theta,Y)
    n=size(Y(:,1),1);
    sigma=zeros(n,1);
    d = zeros(n,1);
    d(1) = 0 ; 
    Y(1) = 0.5;
    sigma(1)= sqrt(sum(Y.^2)/n);
    f = log(1/(sqrt(2*pi)*sigma(1)))-Y(1)^2/(2*sigma(1)^2);
    for i=2:n
        if Y(i-1)<0
            d(i-1)=1;
        else
            d(i-1) = 0;
        end
        sigma(i)=sqrt(theta(1) + theta(2)*Y(i-1)^2+theta(3)*Y(i-1)^2*d(i-1)+theta(4)*sigma(i-1)^2);
        l=log(1/(sqrt(2*pi)*sigma(i)))-Y(i)^2/(2*sigma(i)^2);
        f=f+l;
    end
    f=-f;
end 
