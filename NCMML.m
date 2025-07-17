function [L_best,F2,k,normGrad,ProbFunct] = NCMML(X,label,batchsize,regularization,margins,improve,maxepoch, percent,CVset,Linit,weightimportance)
if isempty(CVset)
    TrainInd = [];
    ValidInd = [];
    TrainPer = percent;
    Trainsize = size(X,1);
else
    TrainInd = CVset{1};
    ValidInd = CVset{2};
    TrainPer = CVset{3};
    Trainsize = length(TrainInd);
end
LabelMatrix = label==unique(label)';
%L = eye(size(X,2));
if isempty(Linit) || all(all(Linit == zeros([margins size(X,2)])))
    L = initialisedW([margins size(X,2)],'xavier');
    L = L/norm(L);
else
    L = Linit;
end
if isempty(weightimportance)
    Wimpt = ones(1,length(unique(label)));
else
    Wimpt = weightimportance;
end
normGrad = [];
%L = initialisedW(size(X,2),'xavier');
if isempty(TrainInd)
    ClassMean=(X.')*LabelMatrix./sum(LabelMatrix);
else
    ClassMean=(X(TrainInd,:).')*LabelMatrix(TrainInd,:)./sum(LabelMatrix(TrainInd,:));
end
P = NCMC(X,L,ClassMean) + eps;
C = P.*LabelMatrix;
Alpha = P - LabelMatrix - eps;
P_i = sum(C,2);
Label_W = LabelMatrix.*Wimpt;
Weight = sum(Label_W ./sum(Label_W,'all'),2);
if isempty(ValidInd)
    F1=sum(Weight.*log(P_i));
else
    F1=sum(Weight(ValidInd).*log(P_i(ValidInd)))./sum(Weight(ValidInd));
end
k=1;
F2=F1;
Eig=eigs(L*L.',1);
valid = 0;
F = F1;
L_best = L;
ProbFunct = @(x) NCMC(x,L_best,ClassMean);
push = 1;
while k <= maxepoch && valid <= improve
    %F = F1;
    batches = ceil(Trainsize/batchsize);
    shuffle = randperm(Trainsize);
    for i = 1:batches
        idx = shuffle((i-1)*batchsize +1 : min(i*batchsize,Trainsize));
        if isempty(TrainInd)
            Grad = NCMMLGradient(X,ClassMean,Alpha,Weight,idx);
        else
            Grad = NCMMLGradient(X,ClassMean,Alpha,Weight,TrainInd(idx));
        end
        if length(F2) > 2 && abs((F2(end) - F2(end-1))/F2(end))< 0.001 && F2(end) < log(0.9*TrainPer)
            %Grad = Grad./norm(Grad);
            push = push + 1;
        else
            push = 1;
        end
        Grad = L*(Grad+regularization);
        normGrad = [normGrad norm(Grad)];
        alpha = (1/2)/length(idx);
        L = L + 0.1*(log(1+push))*Grad;
        %L = L/(trace(L.'*L)^(1/2));
        P = NCMC(X,L,ClassMean) + eps;
        C = P.*LabelMatrix;
        Alpha = P - LabelMatrix - eps;
        P_i = sum(C,2);
        if isempty(ValidInd)
            F1 = sum(Weight.*log(P_i));
            FT = sum(Weight.*log(P_i));
        else
            F1 = sum(Weight(ValidInd).*log(P_i(ValidInd)))./sum(Weight(ValidInd));
            FT = sum(Weight(TrainInd).*log(P_i(TrainInd)))./sum(Weight(TrainInd));
        end
        if F < F1
            F = F1;
            valid = 0;
            L_best = L;
            ProbFunct = @(x) NCMC(x,L_best,ClassMean);
        else
            valid = valid + 1;
        end
        if isnan(F1) || F1 > log(percent) || FT > log(TrainPer) || valid > improve || k > maxepoch
            F2=[F2 F1];
            break
        end
        F2=[F2 F1];
        %plot(1:length(F2)-1,F2(2:end));
        %subplot(2,1,2)
        %plot(1:length(Eig)-1,Eig(2:end));
        %drawnow;
        if isnan(F1) || F1>log(percent) || FT > log(TrainPer) || valid > improve || k > maxepoch
            break
        end
    end
    k=k+1;
    if isnan(F1) || F1>log(percent) || FT > log(TrainPer) || valid > improve || k > maxepoch
        break
    end
end

    
end
function [Grad] = NCMMLGradient(X,ClassMean,Alpha,Weight,Ibatch)

X_ij_all = permute(X(Ibatch,:),[3 2 1]) - ClassMean';
Project = permute((Weight(Ibatch).*Alpha(Ibatch,:))', [1 3 2]);
Grad = sum(pagemtimes(X_ij_all, 'transpose', Project.*X_ij_all,'none'),3)./sum(Weight(Ibatch));

end

function Probability = NCMC(X,L,ClassMean)
LX = L*(X.');
LClassMean = L*ClassMean;
vSsqLX = sum(LX .^ 2, 1);
Exp = -(vSsqLX' - 2*(LX')*LClassMean + sum(LClassMean.^2,1));
Exp = Exp - max(Exp,[],2); 
modi_exp = exp(Exp);
Probability = modi_exp./sum(modi_exp,2);
end
function W = initialisedW(sz,method)
% Glorot initialization
if strcmp(method,'glorot')
    W = (2*rand(sz) - 1) * sqrt(3/sum(sz));
    %W = (W + W')/2;
elseif strcmp(method,'xavier')
    W = (randn(sz)) * sqrt(1/sum(2*sz));
    %W = (W + W')/2;
elseif strcmp(method,'he')
    W = (randn(sz)) * sqrt(2/sum(sz));
    %W = (W + W')/2;
end

end