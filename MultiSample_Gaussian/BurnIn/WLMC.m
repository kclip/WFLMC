function theta_use = WLMC(X,Y,z,S,Sb,a,theta0,K,Nk,l,indc_matrix,eta,K_a)

    theta=theta0; 
    theta_use=[];
    d=length(theta); 
    
    for s=1:S

        Rx=z(:,s); 

        for k=1:K 

            X_local = X((k-1)*Nk+1:k*Nk,:); 
            Y_local = Y((k-1)*Nk+1:k*Nk,:);  
            grad_local= (-(Y_local-X_local*theta)'*X_local)'+theta/K ;

            Rx =  Rx + grad_local* min(1, l/norm(grad_local))*indc_matrix(s,k)*sqrt(a(s)); 
     
        end

        theta=theta-eta*Rx*K/sqrt(a(s))/K_a(s);

        if s>Sb
            theta_use=[theta_use,theta]; 
        end

    end

   

end