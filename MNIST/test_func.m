function [acc, ece] = test_func(theta_use,X_test,Y_test,d)

Su=size(theta_use,2); 
pred_prob=0;
for u=1:Su
    theta_test=reshape(theta_use(:,u),d,10);
    pred_prob=pred_prob+ exp(X_test*theta_test)./sum(exp(X_test*theta_test),2);
end
pred_prob=pred_prob/Su;             
[conf,pred_label]=max(pred_prob');
acc=1-sum(pred_label'~=(Y_test+1))/size(Y_test,1); 

bin_level=linspace(min(conf),1,21); 

ece=0; 
for bin_index=1:20 
sel_sample=find(bin_level(bin_index) <conf<= bin_level(bin_index+1));
ece=ece+length(sel_sample)*abs( mean(conf(sel_sample)) - (1-sum(pred_label(sel_sample)'~=(Y_test(sel_sample)+1))/length(sel_sample) ) ); 
end
ece= ece/size(Y_test,1);