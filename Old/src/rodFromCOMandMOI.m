%% Moment of inertia maker
function [Lp_sol, delta_rho, I_calc, M] = rodFromCOMandMOI(M,I,L,COM_percent,search_bounds,rho_init,search_all,plotting)
close all;
L1  = COM_percent*L
L2  = (1-COM_percent)*L
rho = rho_init.*M/(L)

eqn1 = @(Lp) ( - rho.*(L2.^2-L1.^2)./2)./((Lp.^2-L1.^2)./2);
eqn2 = @(Lp) (I - rho.*(L2.^3+L1.^3)./3)./((Lp.^3+L1.^3)./3);
M_calc = @(Lp, drho) (Lp + L1).*(rho+drho) + (L2-Lp).*rho

Lps = linspace(search_bounds(1),search_bounds(2),1e6);
if search_all
    Lps = linspace(-L1, L2, 1e5);
end

if plotting
figure();
plot(Lps, eqn1(Lps),'b-');
hold on;
plot(Lps, eqn2(Lps),'r-');
end

line_diff = abs(eqn1(Lps)-eqn2(Lps));
if plotting
figure();
plot(Lps, line_diff);
end

Lp_sol = Lps(find(line_diff == min(line_diff),1))
delta_rho = eqn1(Lp_sol)

I_calc   = rho.*(L2.^3 + L1.^3)./3 + delta_rho.*(Lp_sol.^3+L1.^3)./3
COM_calc = rho.*(L2.^2-L1.^2)./2   + delta_rho.*(Lp_sol.^2-L1.^2)./2
M = M_calc(Lp_sol, delta_rho)
end