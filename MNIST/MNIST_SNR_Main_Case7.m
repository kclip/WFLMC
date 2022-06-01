clear all
clc
close all

load('/mnt/WLMC/MNIST/MNIST_Dim30.mat')

rng(10)


N=12000; 
d=size(X_train,2); 

X=[];  %data set 
Y=[];  %labels
%select 1000 samples for each class  
for c=0:9
    index=find(Y_train==c); 
    sel_index=index(randsample(length(index),N/10)); 
    X=[X; X_train(sel_index,:)];
    Y=[Y; Y_train(sel_index)];
end

% label to indicator vector, [0 0 ... 1 ...0] for the non-zero index = c+1
Y_ind= zeros(N,10);
i = [1:N]'; 
j = Y+1;
Y_ind(sub2ind(size(Y_ind), i, j))=1;
clear i j 

K=30; % Num Devices 
Nk=N/K;  % #local data points  
N_0=1;  % Channel noise


L= max(eig(kron(0.5*(eye(10)- 0.1*ones(10,10)),(X'*X) )) )+1; 
mu=1;
eta= 2/(mu+L); %learning rate 1    0.065*2/(mu+L); %learning rate 2    %
gamma=1-eta*mu; 


%clipping 
l=300; 

%model dimension
dm=d*10; 

Sb=550; 
Su=50; 
S=Sb+Su;   %Communication Round
    


%DP requirement 
delta=0.1;
epsilon=50; 
syms x
C(x) =sqrt(pi)*x*exp(x^2); 
invC=finverse(C,x); 

Rdp=(sqrt(epsilon+double(invC(1/delta))^2)-double(invC(1/delta)))^2;



SNR_set= [5:5:40];


acc_Truc_nOP=0;
ece_Truc_nOP=0;
acc_WithoutDP=0;
ece_WithoutDP=0;
acc_Truc_OP=0;
ece_Truc_OP=0;
acc_Truc_OP_het=0;
ece_Truc_OP_het=0;
acc_SBFL=0;
ece_SBFL=0;
acc_NoiselessWithoutDP=0;
ece_NoiselessWithoutDP=0;
acc_NoiselessDP_het=0;
ece_NoiselessDP_het=0;
acc_NoiselessDP=0;
ece_NoiselessDP=0;





for ave_index=1:100
    
    h=0.01*(randn(S,K)+i*randn(S,K))/sqrt(2); 
    h_amp=abs(h); 
    
    theta0= randn(dm,1); 


    z_indv=randn(dm,S,K);

    z=sum(z_indv,3)/sqrt(K); 




    tic
    
    for index=1:length(SNR_set)
    
      
        
        SNR=SNR_set(index);
        P=10^(SNR/10)*N_0*dm; %TX power
        
        %%% OPT trunc threshold %%%
        tt=sort(h_amp,2);  
        g= zeros(S,1);
        grad_error=max(0, N_0*K^2*l^2/P./([K:-1:1].^2)./tt.^2)+4*l^2*(K-[K:-1:1]).^2;
        for s=1:S
           g(s)= tt(s,find(grad_error(s,:)==min(grad_error(s,:)))); 
        end
        indc_matrix=zeros(size(h_amp)); 
        indc_matrix(find(h_amp>=kron(ones(1,K),g)))=1; 
%         indc_matrix=ones(size(h_amp)); 
        K_a=sum(indc_matrix,2); 
        Tx_Time=sum(indc_matrix,1);

        %%% Solution without DP %%%%
        min_a=0;
        min_h=0; 
        for s=1:S
            min_h(s)= min(h_amp(s,h_amp(s,:)>=g(s)));
            min_a(s)=min(min_h(s)^2*P/l^2,eta*N_0*K^2/2/(K_a(s))^2);
        end
        
        a_snr=min_h'.^2*P/l^2;
        a_lmc= eta*N_0*K^2/2./K_a.^2;
        
        min_a=min_a'; 
    
        min_h=min_h'; 

     

        %Scheme3: Optimized threshold and Constant DP at each round 
        a=min(min(N_0*Rdp/2/l^2./Tx_Time),min_a);
        
        theta_Truc_nOP= WLMC(X,Y_ind,z,S,Sb,a,theta0,K,Nk,l,indc_matrix,eta,K_a,X_test,Y_test); 
        [acc, ece] = test_func(theta_Truc_nOP,X_test,Y_test,d);       
        acc_Truc_nOP(ave_index, index)= acc; 
        ece_Truc_nOP(ave_index, index)= ece;  
       
        clear a acc ece theta_Truc_nOP 



        %Scheme4: Truncated Channel inversion Noisy without DP
        a=min_a; 
        
        theta_WithoutDP = WLMC(X,Y_ind,z,S,Sb,a,theta0,K,Nk,l,indc_matrix,eta,K_a,X_test,Y_test); 
        [acc, ece] = test_func(theta_WithoutDP,X_test,Y_test,d);       
        acc_WithoutDP(ave_index, index)= acc; 
        ece_WithoutDP(ave_index, index)= ece;  
       
        clear a acc ece theta_WithoutDP
        
        
        %Scheme2: Optimized threshold and Opt Power Control
        a=  Opt_func_AveDis(min_a,Sb,Su,indc_matrix,Rdp,N_0,l,K_a,gamma); 
        
        theta_Truc_OP = WLMC(X,Y_ind,z,S,Sb,a,theta0,K,Nk,l,indc_matrix,eta,K_a,X_test,Y_test); 
        [acc, ece] = test_func(theta_Truc_OP,X_test,Y_test,d);       
        acc_Truc_OP(ave_index, index)= acc; 
        ece_Truc_OP(ave_index, index)= ece;  
        clear acc ece theta_Truc_OP

        % Heterogeneous Data Setting 
        theta_Truc_OP_het = WLMC_het(X,Y_ind,z,S,Sb,a,theta0,K,Nk,l,indc_matrix,eta,K_a,X_test,Y_test); 
        [acc, ece] = test_func(theta_Truc_OP_het,X_test,Y_test,d);       
        acc_Truc_OP_het(ave_index, index)= acc; 
        ece_Truc_OP_het(ave_index, index)= ece;  
        clear acc ece theta_Truc_OP_het


        % Benchmark SBFL 
        theta_SBFL= SBFL(X,Y_ind,z,S,h_amp,theta0,K,Nk,L,X_test,Y_test,N_0,P);
        [acc, ece] = test_func(theta_SBFL,X_test,Y_test,d);       
        acc_SBFL(ave_index, index)= acc; 
        ece_SBFL(ave_index, index)= ece;  
        clear acc ece theta_SBFL 
            
    end
    
        clear a
     
        %Scheme6: Noiseless without DP -- SGLD
        theta_NoiselessWithoutDP= Noiseless_LMC(X,Y_ind,z,Su,Sb,theta0,eta,X_test,Y_test); 
        [acc, ece] = test_func(theta_NoiselessWithoutDP,X_test,Y_test,d);       
        acc_NoiselessWithoutDP(ave_index)= acc; 
        ece_NoiselessWithoutDP(ave_index)= ece;  
        clear acc ece theta_NoiselessWithoutDP 
             
    
        
        
        %Scheme5: Noiseless with DP
% % % %          min_a= eta*N_0*K^2/2./K_a.^2;
% % % %          a=  Opt_func_AveDis(min_a,Sb,Su,indc_matrix,Rdp,N_0,l,K_a,gamma);       
        min_a=ones(S,1)*eta*N_0/2; 
        a=  Opt_func_AveDis(min_a,Sb,Su,ones(Su+Sb,K),Rdp,N_0,l,ones(S,1)*K,gamma);       
        a=sqrt(N_0./(K*a));

        theta_NoiselessDP= NoiselessDP_LMC(theta0,X,Y_ind,eta,a,S,Sb,K,Nk,z_indv,X_test,Y_test,l);  
        [acc, ece] = test_func(theta_NoiselessDP,X_test,Y_test,d);       
        acc_NoiselessDP(ave_index)= acc;  
        ece_NoiselessDP(ave_index)= ece;  
        clear acc ece  theta_NoiselessDP 
        
        theta_NoiselessDP_het = NoiselessDP_LMC_het(theta0,X,Y_ind,eta,a,S,Sb,K,Nk,z_indv,X_test,Y_test,l);    
        [acc, ece] = test_func(theta_NoiselessDP_het,X_test,Y_test,d);       
        acc_NoiselessDP_het(ave_index)= acc;  
        ece_NoiselessDP_het(ave_index)= ece;   
        clear a  acc ece theta_NoiselessDP_het 
    
   
        ave_index
       
        
        
        figure(1)
        semilogy(SNR_set, mean(acc_Truc_OP,1),'LineWidth',2);
        hold on 
        semilogy(SNR_set, mean(acc_WithoutDP,1), ':','LineWidth',2);
        hold on 
        semilogy(SNR_set, mean(acc_NoiselessWithoutDP)*ones(size(SNR_set)), ':*','LineWidth',2);  
        hold on 
        semilogy(SNR_set, mean(acc_Truc_nOP,1), '--','LineWidth',2);
        hold on 
        semilogy(SNR_set, mean(acc_SBFL,1), '--','LineWidth',2);
        hold off  
    %     legend('Trunc. Opt. PA','Trunc. Unif. DP','Trunc. WithoutDP','Noiseless with DP','Noiseless without DP');
        legend('Adaptive PA','Without DP','Noiseless LMC Without DP','Static PA','SBFL');
        ylabel('Test Accuracy','fontsize',20)
        xlabel('SNR(dB)','fontsize',20)
        set(gca,'FontName','Times New Roman','FontSize',20);   
    %     title('Communication Round S=91')
        grid on 
        pause(0.01)
        
        figure(2)
        semilogy(SNR_set, mean(ece_Truc_OP,1),'LineWidth',2);
        hold on 
        semilogy(SNR_set, mean(ece_WithoutDP,1), ':','LineWidth',2);
        hold on 
        semilogy(SNR_set, mean(ece_NoiselessDP)*ones(size(SNR_set)), '-.','LineWidth',2);  
        hold on 
        semilogy(SNR_set, mean(ece_NoiselessWithoutDP)*ones(size(SNR_set)), ':d','LineWidth',2);  
        hold on 
        semilogy(SNR_set, mean(ece_Truc_nOP,1), '--','LineWidth',2);
        hold on 
        semilogy(SNR_set, mean(ece_SBFL,1), '--*','LineWidth',2);
        hold off  
    %     legend('Trunc. Opt. PA','Trunc. Unif. DP','Trunc. WithoutDP','Noiseless with DP','Noiseless without DP');
        legend('Optimized Power Allocation','Without DP','Noiseless LMC with DP','Noiseless LMC Without DP','Equal Power Allocation','SBFL');
        ylabel('Expected Calibration Error','fontsize',20)
        xlabel('SNR(dB)','fontsize',20)
        set(gca,'FontName','Times New Roman','FontSize',20);   
    %     title('Communication Round S=91')
        grid on 
        pause(0.01)
        

    
    
    
%         figure(3)
%         semilogy(SNR_set, mean(acc_Truc_OP,1),'LineWidth',2);
%         hold on 
%         semilogy(SNR_set, mean(acc_Truc_OP_het,1), '--','LineWidth',2);
%         hold on 
%         semilogy(SNR_set, mean(acc_NoiselessDP)*ones(size(SNR_set)), ':','LineWidth',2);  
%         hold on 
%         semilogy(SNR_set, mean(acc_NoiselessDP_het)*ones(size(SNR_set)), '*','LineWidth',2);
%         hold off  
%     %     legend('Trunc. Opt. PA','Trunc. Unif. DP','Trunc. WithoutDP','Noiseless with DP','Noiseless without DP');
%         legend('Adaptive PA','Adaptive PA (Heterogeneous)','Noiseless LMC With DP','Noiseless LMC With DP (Heterogeneous)');
%         ylabel('Test Accuracy','fontsize',20)
%         xlabel('SNR(dB)','fontsize',20)
%         set(gca,'FontName','Times New Roman','FontSize',20);   
%     %     title('Communication Round S=91')
%         grid on 
%         pause(0.01)
%         
%         figure(4)
%         semilogy(SNR_set, mean(ece_Truc_OP,1),'LineWidth',2);
%         hold on 
%         semilogy(SNR_set, mean(ece_Truc_OP_het,1), '--','LineWidth',2);
%         hold on 
%         semilogy(SNR_set, mean(ece_NoiselessDP)*ones(size(SNR_set)), ':','LineWidth',2);  
%         hold on 
%         semilogy(SNR_set, mean(ece_NoiselessDP_het)*ones(size(SNR_set)), '*','LineWidth',2);
%         hold off  
%     %     legend('Trunc. Opt. PA','Trunc. Unif. DP','Trunc. WithoutDP','Noiseless with DP','Noiseless without DP');
%         legend('Adaptive PA','Adaptive PA (Heterogeneous)','Noiseless LMC With DP','Noiseless LMC With DP (Heterogeneous)');
%         ylabel('Expected Calibration Error','fontsize',20)
%         xlabel('SNR(dB)','fontsize',20)
%         set(gca,'FontName','Times New Roman','FontSize',20);   
%     %     title('Communication Round S=91')
%         grid on 
%         pause(0.01)

   toc 
   
   
   data_name='/mnt/WLMC/MNIST/Result/Varying_SNR7.mat';
                
         save(data_name,'SNR_set','-mat',...
                    'acc_Truc_nOP','-mat',...
                    'ece_Truc_nOP','-mat',...
                    'acc_WithoutDP','-mat',...
                    'ece_WithoutDP','-mat',...
                    'acc_Truc_OP','-mat',...
                    'ece_Truc_OP','-mat',...
                    'acc_Truc_OP_het','-mat',...
                    'ece_Truc_OP_het','-mat',...
                    'acc_SBFL','-mat',...
                    'ece_SBFL','-mat',...
                    'acc_NoiselessWithoutDP','-mat',...
                    'ece_NoiselessWithoutDP','-mat',...
                    'acc_NoiselessDP_het','-mat',...
                    'ece_NoiselessDP_het','-mat',...
                    'acc_NoiselessDP','-mat',...
                    'ece_NoiselessDP','-mat');

end