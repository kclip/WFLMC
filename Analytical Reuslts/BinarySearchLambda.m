function mid_lambda = BinarySearchLambda(ub,lb,lambda,gamma,S,K,eta,N_0,K_a,indc_matrix,min_a,Rdp,l,Lindex)
  
   
    % %UB VALUE 
    % lambda(Subset(1,1))=ub; 
    % curr_a=min([((1+gamma)/2).^(-2*[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);
    % indc_matrix(:,Subset(1,1))'*curr_a'-N_0*Rdp/2/l^2
    % %LB VALUE
    % lambda(Subset(1,1))=lb; 
    % curr_a=min([((1+gamma)/2).^(-2*[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);
    % indc_matrix(:,Subset(1,1))'*curr_a'-N_0*Rdp/2/l^2

     while (ub-lb)/lb >10^(-15)

        mid_lambda=(lb+ub)/2;
        lambda(Lindex)=mid_lambda; 
        curr_a=min([((1+gamma)/2).^(-[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);


        if indc_matrix(:,Lindex)'*curr_a'-N_0*Rdp/2/l^2>0
            lb=mid_lambda;
        else
            ub=mid_lambda;        
        end    

     end
     
     mid_lambda=(lb+ub)/2;
end