function [LambdaRotated,TF,Psi] = BiFactorRotate(Correlation,numF,LBM, UBM)
M = numF+1;

[L,psi,Error] = GetUnrotatedFactors(Correlation,numF,LBM, UBM);
Powers = 2;
while Powers < 30
    [L2,T] = rotatefactors(L,'Method','promax','maxit',500,'power',Powers);
    sortedL = sort(abs(L2),2,'descend');
    if (min(sortedL(:,1)) - max(sortedL(:,2))) > 0
        break
    end
    if Powers == 30
        error('Cannot find oblique structure, pls remove some highly correlated items');
    end
    Powers = Powers + 1;
end
L = L2;
Len = size(Correlation,1);
thres = 0.05;
while ~all(sum(abs(L)>thres,2) == ones(Len,1)) && ~any(sum(abs(L)>thres,2) == zeros(Len,1))
    thres = thres + 0.01;
end

B = (abs(L)>thres);
B = [B ones(Len,1)];
[L,Psi,Error] = GetUnrotatedFactors(Correlation,numF+1,LBM, UBM);
[L,TF] = rotatefactors(L,'Method','procrustes','target',B,'type','orthogonal');
AllPositive = sum(L > 0) == Len;
AllPositive(AllPositive == 0) = 99999;

swap = find(min(AllPositive.*sum(L.^2))); 
T = eye(numF+1);
T(1,:) = zeros(1,numF+1);
T(1,swap) = 1;
T(swap,:) = zeros(1,numF+1);
T(swap,1) = 1;
TF = TF*T;
L = L*T;
Powers = 2;
Diff = Len - 1;
while Powers < 30
    T2 = eye(M);
    [L2,T2(2:end,2:end)] = rotatefactors(L(:,2:end),'Method','promax','maxit',500,'power',Powers);
    sortedL = sort(abs(L2),2,'descend');
    firstnot = find(sum(sortedL(:,1) > sort(sortedL(:,2))')~=Len);
    if isempty(firstnot)
        T2Best = T2;
        break
    end
    Diffnew = Len - firstnot(1);
    if Diffnew < Diff
        Diff = Diffnew;
        T2Best = T2;
    end
    Powers = Powers + 1;
end
L = L*T2Best;
TF = TF*T2Best;
LambdaRotated = L;

end


function [MeanLambda,MeanPsi,Error] = GetUnrotatedFactors(Corr,numF,LowerB, UpperB)
L = zeros([size(Corr,1) numF size(Corr,3)]);
psi = zeros([size(Corr,1) size(Corr,3)]);
Error = zeros(size(Corr,3),1);
Len = size(Corr,1);
randomStart = randn(Len,100);
randomStart = (randomStart.^2);
randomStart = randomStart./max(randomStart);
randomStart(randomStart<0.1) = 0.1;
for i = 1:size(Corr,3)
    d = eig(Corr(:,:,i));
    if issymmetric(Corr(:,:,i)) && all(d > 0)
        Error(i) = sum(Corr(:,:,i) > UpperB,'all') + sum(Corr(:,:,i) < LowerB,'all');
        [L(:,:,i),psi(:,i),~,~] = factoran(Corr(:,:,i),numF,'Xtype','covariance','Nobs',6000,'Rotate','none','Start',randomStart);
    end
end
PsiReject = sum(psi <= 0.006)';
PickLambda = (Error + PsiReject)<1;
MeanLambda = mean(L(:,:,PickLambda),3);
MeanPsi = mean(psi,2);

end