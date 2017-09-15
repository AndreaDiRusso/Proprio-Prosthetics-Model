function F = constraints2(x,M,L,I,COM_percent)

    rho = x(1);
    m1 = x(2);
    m2 = x(3);
        
    L1  = COM_percent*L;
    L2  = (1-COM_percent)*L;
    
    % with two point masses at the ends
    F(1) = rho.*(L2.^2-L1.^2)./2 - m1.*L1 + m2.*L2;
    F(2) = (L2 + L1).*rho + m1 + m2 - M;
    F(3) =  rho.*(L2.^3+L1.^3)./3 + m1.*L1.^2 + m2.*L2.^2 - I;
end