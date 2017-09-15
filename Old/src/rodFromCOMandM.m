%% Moment of inertia maker
function [Lp_sol, delta_rho, I_calc, M] = rodFromCOMandM(M,I,L,COM_percent,search_bounds,rho_init,search_all,plotting)
close all;
L1  = COM_percent*L
L2  = (1-COM_percent)*L
rho = rho_init.*M/(L)

drhofromCOM = @(Lp) ( - rho.*(L2.^2-L1.^2)./2)./((Lp.^2-L1.^2)./2);
drhofromM = @(Lp) ( M - rho*L ) ./( Lp + L1)

Lps = linspace(search_bounds(1),search_bounds(2),1e6);
if search_all
    Lps = linspace(-L1, L2, 1e5);
end

if plotting
figure();
plot(Lps, drhofromCOM(Lps),'b-');
hold on;
plot(Lps, drhofromM(Lps),'r-');
end

line_diff = abs(drhofromCOM(Lps)-drhofromM(Lps));
if plotting
figure();
plot(Lps, line_diff);
end

Lp_sol = Lps(find(line_diff == min(line_diff),1))
delta_rho = drhofromCOM(Lp_sol)

I_calc   = rho.*(L2.^3 + L1.^3)./3 + delta_rho.*(Lp_sol.^3+L1.^3)./3
COM_calc = rho.*(L2.^2-L1.^2)./2   + delta_rho.*(Lp_sol.^2-L1.^2)./2
M_calc = @(Lp, drho) (Lp + L1).*(rho+drho) + (L2-Lp).*rho
M = M_calc(Lp_sol, delta_rho)
end