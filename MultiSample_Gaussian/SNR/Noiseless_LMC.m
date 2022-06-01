 function theta_use = Noiseless_LMC(theta0,X,Y,eta,S,Sb,z)  
    
    theta=theta0; 
    theta_use=[];
    d=size(X,2); 
    
    for s=1:S
        grad= (-(Y-X*theta)'*X)'+theta ;
        
        theta=theta-eta*grad+ sqrt(2*eta)*z(:,s);

        if s>Sb
            theta_use=[theta_use,theta]; 
        end
    end

  
 
 end