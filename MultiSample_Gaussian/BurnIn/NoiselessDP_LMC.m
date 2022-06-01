 function theta_use = NoiselessDP_LMC(theta0,X,Y,eta,a,S,Sb,l,K,Nk)  
    
    theta=theta0; 
    theta_use=[];
    d=size(X,2); 
     for s=1:S

        grad=0; 

        for k=1:K 

            X_local = X((k-1)*Nk+1:k*Nk,:); 
            Y_local = Y((k-1)*Nk+1:k*Nk,:);  
            grad_local= (-(Y_local-X_local*theta)'*X_local)'+theta/K ;
            grad_local= grad_local* min(1, l/norm(grad_local));

            grad=grad+ grad_local+ a(s)*randn(d,1);  
        end

        theta=theta-eta*grad; 

        if s>Sb
            theta_use=[theta_use,theta]; 
        end

    end

 
 end