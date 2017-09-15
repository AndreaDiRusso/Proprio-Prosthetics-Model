function x = make_limb(M,L,I,COM_percent,x0)

if nargin < 5
    x0 = [1,1,1,1];
end
options = optimoptions('fsolve','Algorithm','levenberg-marquardt');
options.MaxIterations = 5e3;
options.MaxFunctionEvaluations = 3e3;

this_problem = @(x) constraints2(x,M,L,I,COM_percent);

x = fsolve(this_problem,x0,options);
lb = [0,0,0];
[x res] = lsqnonlin(this_problem, x0, lb);
fprintf('residual = %4.4f\n',res);

fprintf('rho = %4.4f\n',x(1));
fprintf('m1 = %4.4f\n',x(2));
fprintf('m2 = %4.4f\n',x(3));
%fprintf('m3 = %4.4f\n\n',x(4));

F = constraints2(x,M,L,I,COM_percent);

fprintf('COM = %4.4f\n',F(1));
fprintf('M = %4.4f\n',F(2)+M);
fprintf('I = %4.4f\n\n',F(3)+I);

fprintf('delta_COM = %4.4f\n',F(1).*100./L);
fprintf('delta_M = %4.4f\n',F(2).*100./M);
fprintf('delta_I = %4.4f\n\n',F(3).*100./I);

end