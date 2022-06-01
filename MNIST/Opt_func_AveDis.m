
function P_opt = Opt_func_AveDis(min_a,Sb,Su,indc_matrix,Rdp,N_0,l,K_a,gamma)


if min_a'*indc_matrix*2*l^2/N_0 <= Rdp
	P_opt=min_a;
else
    W=1./K_a(1:Sb).^2.* ( (0.5+0.5*gamma).^(2*Sb-[1:Sb]') )*sum((0.5+0.5*gamma).^(2*[1:Su])); 
    W=[W; 1./K_a(Sb+1:Sb+Su).^2 .* fliplr(cumsum( (0.5+0.5*gamma).^(2*[0:Su-1]) ))' ]; 
     
    
    ub=  max(sqrt(W*2*l^2/N_0)'/Rdp*indc_matrix)^2*2; 
    lb=max(sqrt(W'*N_0/2/l^2)*indc_matrix) /min( min_a'*indc_matrix);
    

    while (ub-lb)/lb >10^(-10)

        mid_lambda=(lb+ub)/2;
        
        curr_P=min([sqrt(N_0*W/2/l^2/mid_lambda), min_a]')';


        if  curr_P'*indc_matrix*2*l^2/N_0 < Rdp
            ub=mid_lambda;
        else
            lb=mid_lambda;        
        end   
        

    end
    
    P_opt=curr_P;
end
     
    