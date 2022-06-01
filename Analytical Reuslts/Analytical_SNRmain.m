clear all
clc
close all

load('Gaussian_Data2.mat')

rng(0)

K=30; %Num Devices 
N_0=1;  %Channel noise
S=50+1;   %Communication Round

tt=svd(inv(Cov_post));
mu=min(tt); %strongly convex 
L=max(tt);  % smoothness 
clear tt 
eta= 0.4/(mu+L); %learning rate 1    0.065*2/(mu+L); %learning rate 2    %
gamma=1-eta*mu; 


l=30;  %clipping 


%DP requirement 
delta=0.01;
epsilon=8; 
syms x
C(x) =sqrt(pi)*x*exp(x^2); 
invC=finverse(C,x); 

Rdp=(sqrt(epsilon+double(invC(1/delta))^2)-double(invC(1/delta)))^2;


%Posterior is Gaussian 
%Initial set as Gaussian 
initi_wd= sum(Mean_post.^2)+trace(Cov_post+pr_sigma^2*eye(d)-2*sqrtm(Cov_post*pr_sigma^2));

SNR=[5:1:40];

Perf_Truc_OP=0;
Perf_Truc_nOP=0;
Perf_Truc_WithoutDP=0;
Perf_NoiselessDP=0;
Perf_NoiselessWithoutDP=0;
  



for ave_index=1
    ave_index
    h=ones(S,K)*0.01;  %(randn(S,K)+i*randn(S,K))/sqrt(2); 
    h_amp=abs(h); 
    
    for snr_index=1:length(SNR)
       
        P=10^(SNR(snr_index)/10)*N_0*d; %TX power

        

% 
%         %Scheme1: Wihtout Truncated threshold and Opt Power Control 
%         
%         g= zeros(S,1);   %ones(S,1)*1.5; %truncated threshold
%         indc_matrix=zeros(size(h_amp)); 
%         indc_matrix(find(h_amp>=kron(ones(1,K),g)))=1; 
%         K_a=sum(indc_matrix,2); 
% 
%         min_a=0;
%         for s=1:S
%             min_h= min(h_amp(s,h_amp(s,:)>=g(s)));
%             min_a(s)=min(min_h^2*P/l^2,eta*N_0*K^2/2/(K_a(s))^2);
%         end
%         
%         Opt_PowerControl2
%         Perf_NTrunc_OP(ave_index,snr_index)= Perf;

        %Scheme3: Optimized threshold and Constant DP at each round        
        %use g(s), indc_matrix, min_a in Scheme2
        clearvars  -except h_amp P l N_0 K SNR ave_index snr_index S Rdp initi_wd eta gamma L mu d Perf_Truc_OP ...
            Perf_Truc_nOP Perf_Truc_WithoutDP Perf_NoiselessDP Perf_Truc_NoiselessWithoutDP Perf_Truc_OP2 Perf_NoiselessWithoutDP
        tt=sort(h_amp,2);  
        g= zeros(S,1);
        grad_error=max(0, N_0*K^2*l^2/P./([K:-1:1].^2)./tt.^2)+4*l^2*(K-[K:-1:1]).^2;
        for s=1:S
           g(s)= tt(s,find(grad_error(s,:)==min(grad_error(s,:)))); 
        end
        indc_matrix=zeros(size(h_amp)); 
        indc_matrix(find(h_amp>=kron(ones(1,K),g)))=1; 
        K_a=sum(indc_matrix,2); 

        min_a=0;
        for s=1:S
            min_h= min(h_amp(s,h_amp(s,:)>=g(s)));
            min_a(s)=min(min_h^2*P/l^2,eta*N_0*K^2/2/(K_a(s))^2);
        end
        
        Tx_Time=sum(indc_matrix,1);
        
        curr_a=min(min(N_0*Rdp/2/l^2./Tx_Time),min_a);

        Perf_Truc_nOP(ave_index,snr_index)=(0.5+0.5*gamma)^(2*S)*initi_wd + (0.5+0.5*gamma).^(2*(S-[1:S]))*(2+2*gamma)/(1-gamma)* ...
                 (eta^4*L^3*d/3+eta^3*L^2*d+4*eta^2*l^2*(K-K_a).^2+ max(0, eta^2*N_0*K^2./(K_a.^2)./curr_a'-2*eta) );
        


        %Scheme2: Optimized threshold and Opt Power Control
        clearvars  -except h_amp P l N_0 K SNR ave_index snr_index S Rdp initi_wd eta gamma L mu d Perf_Truc_OP ...
            Perf_Truc_nOP Perf_Truc_WithoutDP Perf_NoiselessDP Perf_Truc_NoiselessWithoutDP Perf_Truc_OP2 Perf_NoiselessWithoutDP
        tt=sort(h_amp,2);  
        g= zeros(S,1);
        grad_error=max(0, N_0*K^2*l^2/P./([K:-1:1].^2)./tt.^2)+4*l^2*(K-[K:-1:1]).^2;
        for s=1:S
           g(s)= tt(s,find(grad_error(s,:)==min(grad_error(s,:)))); 
        end
        indc_matrix=zeros(size(h_amp)); 
        indc_matrix(find(h_amp>=kron(ones(1,K),g)))=1; 
        K_a=sum(indc_matrix,2); 

        min_a=0;
        for s=1:S
            min_h= min(h_amp(s,h_amp(s,:)>=g(s)));
            min_a(s)=min(min_h^2*P/l^2,eta*N_0*K^2/2/(K_a(s))^2);
        end
        
     
       
        Opt_PowerControl2
        Perf_Truc_OP(ave_index,snr_index)= Perf;
        
        
%     
% 
%         %Scheme2-1: Optimized threshold and Opt Power Control 
%         clearvars  -except h_amp P l N_0 K SNR ave_index snr_index S Rdp initi_wd eta gamma L mu d Perf_Truc_OP ...
%             Perf_Truc_nOP Perf_Truc_WithoutDP Perf_NoiselessDP Perf_Truc_NoiselessWithoutDP Perf_Truc_OP2
%         tt=sort(h_amp,2);  
%         g= zeros(S,1);
%         grad_error=max(0, N_0*K^2*l^2/P./([K:-1:1].^2)./tt.^2)+4*l^2*(K-[K:-1:1]).^2;
%         for s=1:S
%            g(s)= tt(s,find(grad_error(s,:)==min(grad_error(s,:)))); 
%         end
%         indc_matrix=zeros(size(h_amp)); 
%         indc_matrix(find(h_amp>=kron(ones(1,K),g)))=1; 
%         K_a=sum(indc_matrix,2); 
% 
%         min_a=0;
%         for s=1:S
%             min_h= min(h_amp(s,h_amp(s,:)>=g(s)));
%             min_a(s)=min(min_h^2*P/l^2,eta*N_0*K^2/2/(K_a(s))^2);
%         end
%         
%      
%         Opt_PowerControl
%         Perf_Truc_OP(ave_index,snr_index)= Perf;
        
        

        %Scheme4: Truncated Channel inversion Noisy without DP
        clearvars  -except h_amp P l N_0 K SNR ave_index snr_index S Rdp initi_wd eta gamma L mu d Perf_Truc_OP ...
            Perf_Truc_nOP Perf_Truc_WithoutDP Perf_NoiselessDP Perf_Truc_NoiselessWithoutDP Perf_Truc_OP2 Perf_NoiselessWithoutDP
        tt=sort(h_amp,2);  
        g= zeros(S,1);
        grad_error=max(0, N_0*K^2*l^2/P./([K:-1:1].^2)./tt.^2)+4*l^2*(K-[K:-1:1]).^2;
        for s=1:S
           g(s)= tt(s,find(grad_error(s,:)==min(grad_error(s,:)))); 
        end
        indc_matrix=zeros(size(h_amp)); 
        indc_matrix(find(h_amp>=kron(ones(1,K),g)))=1; 
        K_a=sum(indc_matrix,2); 

        min_a=0;
        for s=1:S
            min_h= min(h_amp(s,h_amp(s,:)>=g(s)));
            min_a(s)=min(min_h^2*P/l^2,eta*N_0*K^2/2/(K_a(s))^2);
        end
        
        curr_a=min_a; 
        Perf_Truc_WithoutDP(ave_index,snr_index)=(0.5+0.5*gamma)^(2*S)*initi_wd + (0.5+0.5*gamma).^(2*(S-[1:S]))*(2+2*gamma)/(1-gamma)* ...
                 (eta^4*L^3*d/3+eta^3*L^2*d+4*eta^2*l^2*(K-K_a).^2+ max(0, eta^2*N_0*K^2./(K_a.^2)./curr_a'-2*eta) );

        %Scheme5: Noiseless with DP
        clearvars  -except h_amp P l N_0 K SNR ave_index snr_index S Rdp initi_wd eta gamma L mu d Perf_Truc_OP ...
            Perf_Truc_nOP Perf_Truc_WithoutDP Perf_NoiselessDP Perf_Truc_NoiselessWithoutDP Perf_Truc_OP2 Perf_NoiselessWithoutDP
        indc_matrix=ones(size(h_amp)); 
        K_a=sum(indc_matrix,2); 
        min_a=eta*N_0/2*ones(1,S);
        
        Opt_PowerControl2
        Perf_NoiselessDP(ave_index,snr_index)= Perf;
        
        

        %Scheme6: Noiseless without DP （SGLD）
        clearvars -except h_amp P l N_0 K SNR ave_index snr_index S Rdp initi_wd eta gamma L mu d Perf_Truc_OP ...
            Perf_Truc_nOP Perf_Truc_WithoutDP Perf_NoiselessDP Perf_Truc_NoiselessWithoutDP Perf_Truc_OP2 Perf_NoiselessWithoutDP 
        indc_matrix=ones(size(h_amp)); 
        K_a=sum(indc_matrix,2); 
        min_a=eta*N_0/2*ones(1,S);
        curr_a=min_a; 
        Perf_NoiselessWithoutDP(ave_index,snr_index)=(0.5+0.5*gamma)^(2*S)*initi_wd + (0.5+0.5*gamma).^(2*(S-[1:S]))*(2+2*gamma)/(1-gamma)* ...
                 (eta^4*L^3*d/3+eta^3*L^2*d+4*eta^2*l^2*(K-K_a).^2+ max(0, eta^2*N_0*K^2./(K_a.^2)./curr_a'-2*eta) );
             
             
             
                 
 
    
    end
   
    

% 
% 
%     figure(1)
% %     semilogy(SNR,mean(Perf_NTrunc_OP,1),'--d','LineWidth',2)
% %     hold on 
%     plot(SNR,mean(Perf_Truc_OP,1),'LineWidth',2);
%     hold on 
%     plot(SNR,mean(Perf_Truc_OP2,1),'*','LineWidth',2);
%     hold on 
%     plot(SNR,mean(Perf_Truc_nOP,1),'--','LineWidth',2);
%     hold on 
%     plot(SNR,mean(Perf_Truc_WithoutDP,1),':','LineWidth',2);
%     hold on 
%     plot(SNR,mean(Perf_NoiselessDP,1),'-o','LineWidth',2);
%     hold on 
%     plot(SNR,mean(Perf_Truc_NoiselessWithoutDP,1),':d','LineWidth',2);
%     hold off 
%     legend('Adaptive PA','Adaptive PA2','Static PA','Without DP','Noiseless LMC With DP','Noiseless LMC Without DP');
%     ylabel('Wasserstein distance','fontsize',20)
%     xlabel('SNR(dB)','fontsize',20)
%     set(gca,'FontName','Times New Roman','FontSize',20);    
%     grid on 
%     
     figure(1)
    semilogy(SNR,mean(Perf_Truc_OP,1),'LineWidth',2);
    hold on 
    semilogy(SNR,mean(Perf_Truc_WithoutDP,1),':','LineWidth',2);
    hold on 
    semilogy(SNR([[1:4:36],36]),mean(Perf_NoiselessDP([[1:4:36],36]),1),':d','LineWidth',2);
    hold on 
    semilogy(SNR([[1:4:36],36]),mean(Perf_NoiselessWithoutDP([[1:4:36],36]),1),':*','LineWidth',2);  
    hold on 
    semilogy(SNR,mean(Perf_Truc_nOP,1),'--','LineWidth',2);
    hold off  
%     legend('Trunc. Opt. PA','Trunc. Unif. DP','Trunc. WithoutDP','Noiseless with DP','Noiseless without DP');
    legend('Optimized Power Allocation','Without DP','Noiseless LMC With DP','Noiseless LMC Without DP','Equal Power Allocation');
    ylabel('Wasserstein Distance','fontsize',20)
    xlabel('SNR(dB)','fontsize',20)
    set(gca,'FontName','Times New Roman','FontSize',20);   
%     title('Communication Round S=91')
    grid on 
    pause(0.1)
    
% ylim([10^-3,1*0.5])
legend boxoff





end





