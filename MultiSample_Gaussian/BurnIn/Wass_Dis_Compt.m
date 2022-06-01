function Wass_Dis =  Wass_Dis_Compt(theta_set,Mean_post,Cov_post,burin_set,ave_index,S)

    d=size(theta_set,1)/ave_index; 
    

    
    Su_set=S-burin_set; 
    cumsum_Su=cumsum(Su_set)-Su_set; 
    
    for index=1:length(Su_set) 
        
       for su_index=1: Su_set(index) 
           
            for i=1:ave_index 

                theta_use(1:d, i) = theta_set(d*(i-1)+1: d*i, cumsum_Su(index)+su_index );  
            
            end
            mean_est=mean(theta_use,2); 
            cov_est=(theta_use-mean_est)*(theta_use-mean_est)'/size(theta_use,2); 
            
            clear theta_use

            Wass_Dis_mid(su_index)= sum((mean_est-Mean_post).^2)+ trace(Cov_post+cov_est-2*sqrtm(sqrtm(cov_est)*Cov_post*sqrtm(cov_est)));
       end
       
       Wass_Dis(index)=max( Wass_Dis_mid );
       clear Wass_Dis_mid 

    end
    
end