clear all
clc
close all

load('/mnt/WLMC/Gaussian_Data2.mat')
rng(0)

K=30; % Num Devices 
Nk=N/K;  % #local data points  
N_0=1;  % Channel noise

tt=svd( (X'*X)+ eye(d)/pr_sigma^2); 
mu=min(tt); %strongly convex 
L=max(tt);  % smoothness 
clear tt 
eta= 0.2*2/(mu+L); %learning rate 1    0.065*2/(mu+L); %learning rate 2    %
gamma=1-eta*mu; 






    


%DP requirement 
delta=0.01;
epsilon=15; 
syms x
C(x) =sqrt(pi)*x*exp(x^2); 
invC=finverse(C,x); 

Rdp=(sqrt(epsilon+double(invC(1/delta))^2)-double(invC(1/delta)))^2;


%Posterior is Gaussian 
%Initial set as Gaussian 
initi_wd= sum(Mean_post.^2)+trace(Cov_post+pr_sigma^2*eye(d)-2*sqrtm(Cov_post*pr_sigma^2));



Perf_Truc_OP=0;
Perf_Truc_nOP=0;
Perf_Truc_WithoutDP=0;
Perf_NoiselessDP=0;
Perf_NoiselessWithoutDP=0;


EvlSet_Truc_OP=[];
EvlSet_Truc_nOP=[];
EvlSet_Truc_WithoutDP=[];
EvlSet_NoiselessDP=[];
EvlSet_NoiselessWithoutDP=[];


l=30; 

SNR=30; 
P=10^(SNR/10)*N_0*d; %TX power


burin_set=[10:10:90];   

S=100;   %Communication Rounds 
  

for ave_index=1:100 
    
    
    tic 
    
    
    theta0= randn(d,1); 

    z=randn(d,S);

    h=0.1*(randn(S,K)+i*randn(S,K))/sqrt(2); 
    h_amp=abs(h); 

    %%% OPT trunc threshold %%%
    tt=sort(h_amp,2);  
    g= zeros(S,1);
    grad_error=max(0, N_0*K^2*l^2/P./([K:-1:1].^2)./tt.^2)+4*l^2*(K-[K:-1:1]).^2;

    for s=1:S
        g(s)= tt(s,find(grad_error(s,:)==min(grad_error(s,:)))); 
    end

    indc_matrix=zeros(size(h_amp));
    indc_matrix(find(h_amp>=kron(ones(1,K),g)))=1; 
    K_a=sum(indc_matrix,2); 
    Tx_Time=sum(indc_matrix,1);

    %%% Solution without DP %%%%
    min_a=0;
    min_h=0; 
    for s=1:S
        min_h(s)= min(h_amp(s,h_amp(s,:)>=g(s)));
        min_a(s)=min(min_h(s)^2*P/l^2,eta*N_0*K^2/2/(K_a(s))^2);
    end


    min_a=min_a'; 

    min_h=min_h'; 

    
    EvlSet_Truc_OP_index=[];
    EvlSet_Truc_nOP_index=[];
    EvlSet_Truc_WithoutDP_index=[];
    EvlSet_NoiselessDP_index=[];
    EvlSet_NoiselessWithoutDP_index=[];

    for index=1:length(burin_set)
    
        Sb=burin_set(index); 
        
        Su=S-Sb;  

        
        %Scheme2: Optimized threshold and Opt Power Control

 
        Opt_func

        Ana_Truc_OP(ave_index,index)=cvx_optval; 

        EvlSet_Truc_OP_index=[EvlSet_Truc_OP_index, WLMC(X,Y,z,S,Sb,a,theta0,K,Nk,l,indc_matrix,eta,K_a)]; 

        clear a cvx_optval


        %Scheme3: Optimized threshold and Constant DP at each round 
        a=min(min(N_0*Rdp/2/l^2./Tx_Time),min_a);
        
        EvlSet_Truc_nOP_index= [EvlSet_Truc_nOP_index, WLMC(X,Y,z,S,Sb,a,theta0,K,Nk,l,indc_matrix,eta,K_a)]; 
        
        for s2=Sb+1:S
            Ana_set(s2)=(0.5+0.5*gamma)^(2*s2)*initi_wd + (0.5+0.5*gamma).^(2*(s2-[1:s2]))*(2+2*gamma)/(1-gamma)* ...
                 (eta^4*L^3*d/3+eta^3*L^2*d+4*eta^2*l^2*(K-K_a(1:s2)).^2+ max(0, eta^2*N_0*K^2./(K_a(1:s2).^2).*inv_pos(a(1:s2))-2*eta) ) ;
        end
        
        Ana_Truc_nOP(ave_index,index)=max(Ana_set(Sb+1:S)); 

        clear a  


        %Scheme4: Truncated Channel inversion Noisy without DP
        a=min_a; 
        
        EvlSet_Truc_WithoutDP_index=[EvlSet_Truc_WithoutDP_index,  WLMC(X,Y,z,S,Sb,a,theta0,K,Nk,l,indc_matrix,eta,K_a)]; 
       
        for s2=Sb+1:S
            Ana_set(s2)=(0.5+0.5*gamma)^(2*s2)*initi_wd + (0.5+0.5*gamma).^(2*(s2-[1:s2]))*(2+2*gamma)/(1-gamma)* ...
                 (eta^4*L^3*d/3+eta^3*L^2*d+4*eta^2*l^2*(K-K_a(1:s2)).^2+ max(0, eta^2*N_0*K^2./(K_a(1:s2).^2).*inv_pos(a(1:s2))-2*eta) ) ;
        end
        
        Ana_Truc_WithoutDP(ave_index,index)=max(Ana_set(Sb+1:S)); 

        clear a  
      

    
        %Scheme5: Noiseless with DP
   
        Opt_Noiseless_Func 
        
        Ana_NoiselessDP(ave_index,index)=cvx_optval; 

        a=sqrt(N_0./(K*a));

        EvlSet_NoiselessDP_index=[EvlSet_NoiselessDP_index,NoiselessDP_LMC(theta0,X,Y,eta,a,S,Sb,l,K,Nk)];        
         
     

        clear a cvx_optval 


        %Scheme6: Noiseless without DP （SGLD）
        EvlSet_NoiselessWithoutDP_index= [EvlSet_NoiselessWithoutDP_index, Noiseless_LMC(theta0,X,Y,eta,S,Sb,z)] ; 
        
        for s2=Sb+1:S
            Ana_set(s2)=(0.5+0.5*gamma)^(2*s2)*initi_wd + sum( (0.5+0.5*gamma).^(2*(s2-[1:s2]))*(2+2*gamma)/(1-gamma)* ...
                 (eta^4*L^3*d/3+eta^3*L^2*d) );
        end
        
        Ana_NoiselessWithoutDP(ave_index,index)=max(Ana_set(Sb+1:S)); 
             
        clear a
    end
    
    EvlSet_Truc_OP=[EvlSet_Truc_OP; EvlSet_Truc_OP_index];
    EvlSet_Truc_nOP=[EvlSet_Truc_nOP; EvlSet_Truc_nOP_index]; 
    EvlSet_Truc_WithoutDP= [EvlSet_Truc_WithoutDP; EvlSet_Truc_WithoutDP_index]; 
    EvlSet_NoiselessDP = [EvlSet_NoiselessDP; EvlSet_NoiselessDP_index]; 
    EvlSet_NoiselessWithoutDP = [EvlSet_NoiselessWithoutDP; EvlSet_NoiselessWithoutDP_index]; 
    
    clear  EvlSet_Truc_OP_index EvlSet_Truc_nOP_index EvlSet_Truc_WithoutDP_index EvlSet_NoiselessDP_index EvlSet_NoiselessWithoutDP_index 

    
    
     ave_index
    
    if  ave_index >= 10
        Perf_Truc_OP= Wass_Dis_Compt(EvlSet_Truc_OP,Mean_post,Cov_post,burin_set,ave_index,S)
        Perf_Truc_WithoutDP= Wass_Dis_Compt(EvlSet_Truc_WithoutDP,Mean_post,Cov_post,burin_set,ave_index,S);
        Perf_NoiselessDP = Wass_Dis_Compt(EvlSet_NoiselessDP,Mean_post,Cov_post,burin_set,ave_index,S);
        Perf_NoiselessWithoutDP =  Wass_Dis_Compt(EvlSet_NoiselessWithoutDP,Mean_post,Cov_post,burin_set,ave_index,S);
        Perf_Truc_nOP =  Wass_Dis_Compt(EvlSet_Truc_nOP,Mean_post,Cov_post,burin_set,ave_index,S);

    
        figure(1)
        semilogy(burin_set, Perf_Truc_OP,'LineWidth',2);
        hold on 
        semilogy(burin_set, Perf_Truc_WithoutDP, ':','LineWidth',2);
        hold on 
        semilogy(burin_set, Perf_NoiselessDP, ':d','LineWidth',2);
        hold on 
        semilogy(burin_set, Perf_NoiselessWithoutDP, ':*','LineWidth',2);  
        hold on 
        semilogy(burin_set, Perf_Truc_nOP, '--','LineWidth',2);
        hold off  
    %     legend('Trunc. Opt. PA','Trunc. Unif. DP','Trunc. WithoutDP','Noiseless with DP','Noiseless without DP');
        legend('Adaptive PA','Without DP','Noiseless LMC With DP','Noiseless LMC Without DP','Static PA');
        ylabel('Wasserstein Distance','fontsize',20)
        xlabel('Burn-in Period','fontsize',20)
        set(gca,'FontName','Times New Roman','FontSize',20);   
    %     title('Communication Round S=91')
        grid on 
        pause(0.01)

    
    
    
        figure(2)
        semilogy(burin_set,mean(Ana_Truc_OP,1),'LineWidth',2);
        hold on 
        semilogy(burin_set,mean(Ana_Truc_WithoutDP,1),':','LineWidth',2);
        hold on 
        semilogy(burin_set,mean(Ana_NoiselessDP,1),':d','LineWidth',2);
        hold on 
        semilogy(burin_set,mean(Ana_NoiselessWithoutDP,1),':*','LineWidth',2);  
        hold on 
        semilogy(burin_set,mean(Ana_Truc_nOP,1),'--','LineWidth',2);
        hold off  
    %     legend('Trunc. Opt. PA','Trunc. Unif. DP','Trunc. WithoutDP','Noiseless with DP','Noiseless without DP');
        legend('Adaptive PA','Without DP','Noiseless LMC With DP','Noiseless LMC Without DP','Static PA');
        ylabel('Wasserstein Distance Ana','fontsize',20)
        xlabel('Burn-in Period','fontsize',20)
        set(gca,'FontName','Times New Roman','FontSize',20);   
    %     title('Communication Round S=91')
        grid on 
        pause(0.01)
    
    end
    
    
    
% ylim([10^-3,1*0.5])
% legend boxoff

   toc 
end


