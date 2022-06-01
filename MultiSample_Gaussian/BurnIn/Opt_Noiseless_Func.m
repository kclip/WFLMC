
%%%%  CVX  %%%%%
cvx_begin  quiet 
   variables a(S,1)  nu  
   minimize nu  
   subject to
      ones(1,S)*2*a*l^2/N_0 <= Rdp;

       a- eta*N_0/2 <= 0 ; 
       
       for s2=Sb+1:S
       (0.5+0.5*gamma)^(2*s2)*initi_wd + (0.5+0.5*gamma).^(2*(s2-[1:s2]))*(2+2*gamma)/(1-gamma)* ...
                 (eta^4*L^3*d/3+eta^3*L^2*d + max(0, eta^2*N_0*inv_pos(a(1:s2))-2*eta) ) <= nu ;
       end
       
       a >= 0
cvx_end