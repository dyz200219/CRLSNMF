clear;
clc;
addpath('tool/', 'dataset/', 'user');

load handwritten.mat
X{1} = pixel'; 
X{2} = fourier';
X{3} = zer';
X{4} = profile';
X{5} = mor';
truth = gnd + 1; 

maxiter = 200;
class_num = length(unique(truth)); 

beta = 1e6;
gamma = 1e8;
sigma = 1e3;     % sigma = alpha/2

fileID = fopen('handwritten_MYNMF_tune.txt','a');
replic = 10;

AC_ = zeros(1, replic);
NMI_ = zeros(1, replic);
purity_ = zeros(1, replic);
Fscore_ = zeros(1, replic);
Precision_ = zeros(1, replic);
Recall_ = zeros(1, replic);
AR_ = zeros(1, replic);

for i = 1: replic
    [V_star, obj,V0] = CRLSNMF_update(X,truth, 'beta', beta, 'gamma', gamma, 'sigma', sigma, 'maxiter', maxiter);
    idx = litekmeans(V_star', class_num, 'Replicates', 20); %kmeans
    result = EvaluationMetrics(truth, idx);
    AC_(i) = result(1);
    NMI_(i) = result(2);
    purity_(i) = result(3);
    Fscore_(i) = result(4);
    Precision_(i) = result(5);
    Recall_(i) = result(6);
    AR_(i) = result(7);
end

AC(1) = mean(AC_); AC(2) = std(AC_);
NMI(1) = mean(NMI_); NMI(2) = std(NMI_);
purity(1) = mean(purity_); purity(2) = std(purity_);
Fscore(1) = mean(Fscore_); Fscore(2) = std(Fscore_);
Precision(1) = mean(Precision_); Precision(2) = std(Precision_);
Recall(1) = mean(Recall_); Recall(2) = std(Recall_);
AR(1) = mean(AR_); AR(2) = std(AR_);
fprintf(fileID, "alpha = %g, beta = %g, lamaga = %g :\n", beta, gamma, sigma);
fprintf(fileID, "AC = %5.4f + %5.4f, NMI = %5.4f + %5.4f, purity = %5.4f + %5.4f\nFscore = %5.4f + %5.4f, Precision = %5.4f + %5.4f, Recall = %5.4f + %5.4f, AR = %5.4f + %5.4f\n",...
    AC(1), AC(2), NMI(1), NMI(2), purity(1), purity(2), Fscore(1), Fscore(2), Precision(1), Precision(2), Recall(1), Recall(2), AR(1), AR(2));
fprintf(fileID,'********************************\n');
fclose(fileID);

plot(obj);





