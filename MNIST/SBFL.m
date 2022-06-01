function theta_use = SBFL(X,Y_ind,z,S,h_amp,theta0,K,Nk,L,X_test,Y_test,N_0,P)

    theta=theta0; 
    theta_use=[];
    theta_plot=[]; 
    dm=length(theta); 
    d=dm/10; 
    
    eta_scal= 0.0497;   %1/(mu+L/N); 
    delta=0.9; 
    
    sel_index= reshape([1:Nk*K], Nk*K/10, 10); 
    
    mid_term=0; 
    
    for s=1:floor(S/K)


        theta_matrix= reshape(theta,d,10); 

        update_term=0; 
        
        for k=1:K 
            
            % homogeneous setting 
            local_index= sel_index(Nk/10*(k-1)+1: Nk/10*k,:); 
            local_index= reshape(local_index,Nk,1); 
            local_data=X(local_index,:); 
            local_label=Y_ind(local_index,:);
            
           
            
            grad_local= reshape(sum( kron(ones(1, 10 ), local_data) .* kron( ( exp(local_data*theta_matrix)./sum(exp(local_data*theta_matrix),2)  - local_label ) ,ones(1,d) ) ,1),d,10)+theta_matrix/K  ;
            grad_local= reshape(grad_local,dm,1)/Nk; 
            
            mean_gk=mean(grad_local); 
            var_gk= sqrt(mean(grad_local.^2)-mean(grad_local)^2); 
            grad_local=grad_local- mean(grad_local);  
            
            yk=h_amp((s-1)*K+k,k)*sign(grad_local)+ z(:,(s-1)*K+k);
            
            update_term=update_term+ mean_gk*ones(dm,1)+ sqrt(2/pi)*var_gk*tanh(2*h_amp((s-1)*K+k,k)*yk/(N_0/P)); 
     
        end
        mid_term=delta*mid_term+ update_term; 
        theta=theta-eta_scal*mid_term;
        
        
% % % % % %%%%%%% Convergence Test Plot %%%%%%% 
% % % % %         [acc, ece] = test_func(theta,X_test,Y_test,d);               
% % % % %         test_acc(s)=acc; 
% % % % %         ece_set(s)=ece; 
% % % % % 
% % % % %         figure(1)
% % % % %         plot(test_acc(find(test_acc>0)))
% % % % %         figure(2) 
% % % % %         plot(ece_set(find(ece_set>0)))
% % % % %         pause(0.1)

         
        

    end
    
    theta_use=theta; 

   

end