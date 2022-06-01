function Wass_Dis =  Wass_Dis_Compt(theta_set,Mean_post,Cov_post)

    d=size(theta_set,1); 
    ave_index=size(theta_set,3); 
    
    Wass_Dis=[]; 
    for su_index=1: size(theta_set,2)  

        for index= 1: size(theta_set,4)
 
            theta_use(1:d, 1:ave_index) = theta_set(:,su_index,:, index); 

            mean_est=mean(theta_use,2); 
            cov_est=(theta_use-mean_est)*(theta_use-mean_est)'/size(theta_use,2); 

            Wass_Dis(index,su_index)= sum((mean_est-Mean_post).^2)+ trace(Cov_post+cov_est-2*sqrtm(sqrtm(cov_est)*Cov_post*sqrtm(cov_est)));
        end
    end
    
   Wass_Dis= max(Wass_Dis'); 
    
 end