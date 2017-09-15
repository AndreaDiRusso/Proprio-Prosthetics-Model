clear;clc;close all;

guesses = 0.6:0.1:10;
Is = []
for a = 1:length(guesses)
    [Lp_sol, delta_rho, I_calc, M] = ...
        rodFromCOMandM(101,1.21e5,74,0.62,[55.6 55.7],guesses(a),1,0)
    Is = [Is I_calc];
end

figure();
plot(guesses,Is)
    
    