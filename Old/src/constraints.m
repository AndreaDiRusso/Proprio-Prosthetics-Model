function F = constraints(x)

    rho1 = x(1);
    rho2 = x(2);
    Lp = x(3);
    m1 = x(4);
    %m2 = x(5);
    
    M = 557;
    L = 163;
    COM_percent = 0.41;
    I = 1.61e6;
    
    L1  = COM_percent*L;
    L2  = (1-COM_percent)*L;
    
% with two point masses at the ends
%     F(1) = rho1.*(Lp.^2-L1.^2)./2 + rho2.*(L2.^2 - Lp.^2)./2 - m1.*L1 + m2.*L2;
%     F(2) = (Lp + L1).*rho1 + (L2 - Lp).*rho2 + m1 + m2 - M;
%     F(3) =  rho1.*(Lp.^3+L1.^3)./3 + rho2.*(L2.^3 - Lp.^3)./3 + m1.*L1.^2 + m2.*L2.^2 - I;

% with a point mass at the top
    F(1) = rho1.*(Lp.^2-L1.^2)./2 + rho2.*(L2.^2 - Lp.^2)./2 - m1.*L1;
    F(2) = (Lp + L1).*rho1 + (L2 - Lp).*rho2 + m1 - M;
    F(3) =  rho1.*(Lp.^3+L1.^3)./3 + rho2.*(L2.^3 - Lp.^3)./3 + m1.*L1.^2 - I;
    
end