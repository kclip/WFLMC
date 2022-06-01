 function theta_use = Noiseless_LMC(X,Y_ind,z,Su,Sb,theta0,eta,X_test,Y_test)
    
    theta=theta0; 
    theta_use=[];
    theta_plot=[]; 
    
    dm=length(theta); 
    d=dm/10; 
    
    S=Su+Sb; 
    for s=1:S
        
        theta_matrix= reshape(theta,d,10); 
        grad = reshape(sum(kron(ones(1, 10 ), X) .* kron( ( exp(X*theta_matrix)./sum(exp(X*theta_matrix),2)  - Y_ind ) ,ones(1,d) ) ,1),d,10)+theta_matrix  ;
              
        theta=theta-eta*reshape(grad,dm,1)+ sqrt(2*eta)*z(:,s);

        if s>Sb
            theta_use=[theta_use,theta]; 
        end

    
% % % %%%%%%% Convergence Test Plot %%%%%%% 
% % %         theta_plot=[theta_plot, theta];
% % %         
% % %         if s>=Su 
% % %             if mod(s,50)==0 
% % %                
% % %                 [acc, ece] = test_func(theta_plot(:,s-Su+1:s),X_test,Y_test,d);               
% % %                 test_acc(s-Su+1)=acc; 
% % %                 ece_set(s-Su+1)=ece; 
% % %                 
% % %                 figure(1)
% % %                 plot(test_acc(find(test_acc>0)))
% % %                 figure(2) 
% % %                 plot(ece_set(find(ece_set>0)))
% % %                 pause(0.1)
% % %             end
% % %             
% % %         end 
% % %         
% % %     
    end

  
 
 end