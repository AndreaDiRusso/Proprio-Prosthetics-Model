clear;clc;close all;

x0 = [1,1,1];
options = optimoptions('fsolve');
options.MaxIterations = 5e3;
options.MaxFunctionEvaluations = 3e3;

M = 557;
L = 163;
COM_percent = 0.41;
I = 1.61e6;

this_problem = @(x) constraints2(x,M,L,I,COM_percent);

x = fsolve(this_problem,x0,options);

fprintf('rho = %4.4f\n',x(1));
fprintf('m1 = %4.4f\n',x(2));
fprintf('m2 = %4.4f\n',x(3));