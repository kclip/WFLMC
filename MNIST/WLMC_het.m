function theta_use = WLMC_het(X,Y_ind,z,S,Sb,a,theta0,K,Nk,l,indc_matrix,eta,K_a,X_test,Y_test)

    theta=theta0; 
    theta_use=[];
    theta_plot=[]; 
    
    dm=length(theta); 
    d=dm/10; 
    
    Su=S-Sb; 
      
    
    for s=1:S

        Rx=z(:,s); 
        
        theta_matrix= reshape(theta,d,10); 

        for k=1:K 
            
            % heterogeneous setting 
           
            local_data=X(Nk*(k-1)+1:Nk*k,:); 
            local_label=Y_ind(Nk*(k-1)+1:Nk*k,:);
          
            
            grad_local= reshape(sum(kron(ones(1, 10 ), local_data) .* kron( ( exp(local_data*theta_matrix)./sum(exp(local_data*theta_matrix),2)  - local_label ) ,ones(1,d) ) ,1),d,10)+theta_matrix/K  ;
            grad_local= reshape(grad_local,dm,1); 
            Rx =  Rx + grad_local* min(1, l/norm(grad_local))*indc_matrix(s,k)*sqrt(a(s)); 
     
        end

        theta=theta-eta*Rx*K/sqrt(a(s))/K_a(s);

%%%%%%% Samples for use %%%%%%%         
        if s>Sb
            theta_use=[theta_use,theta]; 
        end

% % % % %%%%%%% Convergence Test Plot %%%%%%% 
% % % %         theta_plot=[theta_plot, theta];
% % % %         
% % % %         if s>=Su 
% % % %             if mod(s,50)==0 
% % % %                
% % % %                 [acc, ece] = test_func(theta_plot(:,s-Su+1:s),X_test,Y_test,d);               
% % % %                 test_acc(s-Su+1)=acc; 
% % % %                 ece_set(s-Su+1)=ece; 
% % % %                 
% % % %                 figure(1)
% % % %                 plot(test_acc(find(test_acc>0)))
% % % %                 figure(2) 
% % % %                 plot(ece_set(find(ece_set>0)))
% % % %                 pause(0.1)
% % % %             end
% % % %             
% % % %         end 
        

    end

   

end