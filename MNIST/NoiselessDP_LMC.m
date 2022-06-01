 function theta_use = NoiselessDP_LMC(theta0,X,Y_ind,eta,a,S,Sb,K,Nk,z_indv,X_test,Y_test,l)
    
    theta=theta0; 
    theta_use=[];
    theta_plot=[]; 
    
    dm=length(theta); 
    d=dm/10; 
    
    Su=S-Sb; 
    
    sel_index= reshape([1:Nk*K], Nk*K/10, 10); 
    
     for s=1:S

        grad=0; 

        theta_matrix= reshape(theta,d,10); 
        
        for k=1:K 

             % homogeneous setting 
            local_index= sel_index(Nk/10*(k-1)+1: Nk/10*k,:); 
            local_index= reshape(local_index,Nk,1); 
            local_data=X(local_index,:); 
            local_label=Y_ind(local_index,:);
            
            
            grad_local= reshape(sum(kron(ones(1, 10 ), local_data) .* kron( ( exp(local_data*theta_matrix)./sum(exp(local_data*theta_matrix),2)  - local_label ) ,ones(1,d) ) ,1),d,10)+theta_matrix/K  ;
            grad_local= reshape(grad_local,dm,1); 
            
            
           
            

            grad=grad+  grad_local* min(1, l/norm(grad_local)) + a(s)*z_indv(:,s,k);  
        end

        theta=theta-eta*grad; 

        if s>Sb
            theta_use=[theta_use,theta]; 
        end
        
% % % % % % %%%%%%% Convergence Test Plot %%%%%%% 
% % % % % %         theta_plot=[theta_plot, theta];
% % % % % %         
% % % % % %         if s>=Su 
% % % % % %             if mod(s,50)==0 
% % % % % %                
% % % % % %                 [acc, ece] = test_func(theta_plot(:,s-Su+1:s),X_test,Y_test,d);               
% % % % % %                 test_acc(s-Su+1)=acc; 
% % % % % %                 ece_set(s-Su+1)=ece; 
% % % % % %                 
% % % % % %                 figure(1)
% % % % % %                 plot(test_acc(find(test_acc>0)))
% % % % % %                 figure(2) 
% % % % % %                 plot(ece_set(find(ece_set>0)))
% % % % % %                 pause(0.1)
% % % % % %             end
% % % % % %             
% % % % % %         end 
% % % % % %         
    end

 
 end