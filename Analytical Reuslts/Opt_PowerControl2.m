OverDP=indc_matrix'*min_a'-N_0*Rdp/2/l^2;
Subset=[find(OverDP>0),OverDP(find(OverDP>0))];
Subset=sortrows(Subset,-2);
lambda=zeros(1,K);

OP_index=[]; %index of operating set

while size(Subset,1)>0
    

    OP_index=[OP_index,Subset(1,1)]; 
    ub=(((1+gamma)/2)^(-S)*K*eta*sqrt(N_0)*S/(N_0*Rdp/2/l^2))^2;  %set_lambda_max(find(set_lambda_max(:,1)==NZ_index(i)),2); 
    lb=0;
    lambda(Subset(1,1))=ub; 
    curr_a=min([((1+gamma)/2).^(-[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);
    while  indc_matrix(:,Subset(1,1))'*curr_a'-N_0*Rdp/2/l^2>0
        lb=ub;
        ub=ub*2;
        lambda(Subset(1,1))=ub; 
        curr_a=min([((1+gamma)/2).^(-[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);
    end
    lambda(Subset(1,1))= BinarySearchLambda(ub,lb,lambda,gamma,S,K,eta,N_0,K_a,indc_matrix,min_a,Rdp,l,Subset(1,1));  
    curr_a=min([((1+gamma)/2).^(-[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);
    OverDP=indc_matrix'*curr_a'-N_0*Rdp/2/l^2;
    
        
    while sum(abs(OverDP(find(lambda>0)))>1e-10 )>0

        for i=1:length(OP_index) 
 
            lambda(OP_index(i))=0;             
            curr_a=min([((1+gamma)/2).^(-[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);
            OverDP=indc_matrix'*curr_a'-N_0*Rdp/2/l^2;
            
            if OverDP(OP_index(i))>0                
             
                ub=(((1+gamma)/2)^(-S)*K*eta*sqrt(N_0)*S/(N_0*Rdp/2/l^2))^2;  %set_lambda_max(find(set_lambda_max(:,1)==NZ_index(i)),2); 
                lb=0;
                lambda(OP_index(i))=ub; 
                curr_a=min([((1+gamma)/2).^(-[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);
                while  indc_matrix(:,OP_index(i))'*curr_a'-N_0*Rdp/2/l^2>0
                    lb=ub;
                    ub=ub*2;
                    lambda(OP_index(i))=ub; 
                    curr_a=min([((1+gamma)/2).^(-[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);
                end
                lambda(OP_index(i))= BinarySearchLambda(ub,lb,lambda,gamma,S,K,eta,N_0,K_a,indc_matrix,min_a,Rdp,l,OP_index(i));      
           
            end
            
            curr_a=min([((1+gamma)/2).^(-[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);
            OverDP=indc_matrix'*curr_a'-N_0*Rdp/2/l^2;


        end
        
       
    end
    

    Subset=[find(OverDP>10^(-10)),OverDP(find(OverDP>10^(-10)))];
    Subset=sortrows(Subset,-2);
end
    curr_a=min([((1+gamma)/2).^(-[1:S])*K*eta*sqrt(N_0)./K_a'./sqrt(indc_matrix*lambda')'; min_a]);
    Perf=(0.5+0.5*gamma)^(2*S)*initi_wd + (0.5+0.5*gamma).^(2*(S-[1:S]))*(2+2*gamma)/(1-gamma)* ...
                 (eta^4*L^3*d/3+eta^3*L^2*d+4*eta^2*l^2*(K-K_a).^2+ max(0, eta^2*N_0*K^2./(K_a.^2)./curr_a'-2*eta) );