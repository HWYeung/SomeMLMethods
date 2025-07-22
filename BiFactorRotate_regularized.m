function [LambdaRotated,TF,Psi] = BiFactorRotate_regularized(Correlation,numF,LBM,UBM,lambda,penalty)
% BiFactorRotate_regularized - Performs penalized bifactor rotation with structure checks.
%
% Syntax:
%   [LambdaRotated,TF,Psi] = BiFactorRotate_regularized(Correlation,numF,LBM,UBM,lambda,penalty)
%
% Inputs:
%   Correlation - [p x p x N] array of correlation matrices from bootstrapped samples
%   numF        - Number of group (specific) factors
%   LBM         - Lower bound matrix for correlation thresholds (default: zeros)
%   UBM         - Upper bound matrix for correlation thresholds (default: ones)
%   lambda      - Regularization strength (default: 1e-4)
%   penalty     - Type of regularization: 'L2', 'L1', or 'EN' (elastic net) (default: 'L2')
%
% Outputs:
%   LambdaRotated - Final rotated loading matrix [p x (numF+1)]
%   TF            - Final transformation matrix applied during rotation
%   Psi           - Final uniqueness estimates

if nargin < 3 || isempty(LBM), LBM = 0; end
if nargin < 4 || isempty(UBM), UBM = 1; end
if nargin < 5 || isempty(lambda), lambda = 1e-4; end
if nargin < 6 || isempty(penalty), penalty = 'L2'; end

M = numF+1;

% Step 1: Estimate unrotated factor loadings using EM with regularization
[L,psi,Error] = GetUnrotatedFactors_regularized(Correlation,numF,LBM,UBM,lambda,penalty);

% Step 2: Attempt Promax rotation to find a clean structure
Powers = 2;
while Powers < 30
    [L2,T] = rotatefactors(L,'Method','promax','maxit',500,'power',Powers);
    sortedL = sort(abs(L2),2,'descend');
    if (min(sortedL(:,1)) - max(sortedL(:,2))) > 0
        break
    end
    if Powers == 30
        error('Cannot find oblique structure, please remove some highly correlated items');
    end
    Powers = Powers + 1;
end
L = L2;

% Step 3: Determine binary structure mask based on thresholding
Len = size(Correlation,1);
thres = 0.05;
while ~all(sum(abs(L)>thres,2) == 1) && ~any(sum(abs(L)>thres,2) == 0)
    thres = thres + 0.01;
end

% Step 4: Construct bifactor binary mask (1 general + specific factors)
B = (abs(L)>thres);
B = [B ones(Len,1)];

% Step 5: Re-estimate with full (general + group) factors
[L,Psi,Error] = GetUnrotatedFactors_regularized(Correlation,numF+1,LBM,UBM,lambda,penalty);

% Step 6: Procrustes rotation to bifactor structure
[L,TF] = rotatefactors(L,'Method','procrustes','target',B,'type','orthogonal');

% Step 7: Make sure the general factor is the first column
AllPositive = sum(L > 0) == Len;
AllPositive(AllPositive == 0) = 99999;
swap = find(min(AllPositive.*sum(L.^2))); 
T = eye(numF+1);
T(1,:) = 0;
T(1,swap) = 1;
T(swap,:) = 0;
T(swap,1) = 1;
TF = TF*T;
L = L*T;

% Step 8: Optional Promax rotation of group (specific) factors only
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

% Step 9: Final rotation application
L = L*T2Best;
TF = TF*T2Best;
LambdaRotated = L;
end


function [MeanLambda,MeanPsi,Error] = GetUnrotatedFactors_regularized(Corr,numF,LowerB,UpperB,lambda,penalty)
% GetUnrotatedFactors_regularized - Estimate unrotated loadings with penalized EM over bootstrap samples
%
% Inputs:
%   Corr   - [p x p x N] bootstrap correlation matrices
%   numF   - number of factors to extract
%   LowerB - lower bound matrix for correlation thresholds
%   UpperB - upper bound matrix for correlation thresholds
%   lambda - regularization strength
%   penalty- 'L2', 'L1', or 'EN' for elastic net
%
% Outputs:
%   MeanLambda - mean factor loadings over acceptable bootstrap samples
%   MeanPsi    - mean uniqueness estimates
%   Error      - vector of error metrics per bootstrap

L = zeros(size(Corr,1),numF,size(Corr,3));
psi = zeros(size(Corr,1),size(Corr,3));
Error = zeros(size(Corr,3),1);
Len = size(Corr,1);

randomStart = randn(Len,100);
randomStart = (randomStart.^2);
randomStart = randomStart./max(randomStart);
randomStart(randomStart<0.1) = 0.1;

for i = 1:size(Corr,3)
    d = eig(Corr(:,:,i));
    if issymmetric(Corr(:,:,i)) && all(d > 0)
        % Penalize correlation matrix entries outside acceptable bounds
        Error(i) = sum(Corr(:,:,i) > UpperB,'all') + sum(Corr(:,:,i) < LowerB,'all');
        [L(:,:,i),psi(:,i)] = EM_penalized_factor(Corr(:,:,i),numF,lambda,penalty,randomStart);
    end
end

% Reject bootstraps with uniqueness too small or error too high
PsiReject = sum(psi <= 0.006)';
PickLambda = (Error + PsiReject) < 1;

MeanLambda = mean(L(:,:,PickLambda),3);
MeanPsi = mean(psi(:,PickLambda),2);
end

function [Lambda, Psi] = EM_penalized_factor(S, numF, lambda, penalty, initLambda)
% EM_penalized_factor - Penalized EM algorithm for factor analysis.
%
% Inputs:
%   S         - Covariance or correlation matrix (p x p)
%   numF      - Number of factors to extract
%   lambda    - Regularization strength
%   penalty   - Type of penalty: 'L2', 'L1', or 'EN'
%   initLambda - Initial loading guesses (p x nInit)
%
% Outputs:
%   Lambda    - Factor loading matrix (p x numF)
%   Psi       - Uniqueness vector (p x 1)

[p, ~] = size(S);
Psi = diag(S);
Lambda = initLambda(:,1:numF);
iters = 100;
for iter = 1:iters
    invPsi = diag(1./Psi);
    M = Lambda' * invPsi * Lambda + eye(numF);
    invM = inv(M);
    Ez = invM * Lambda' * invPsi * S;
    Ezz = invM + Ez * Ez';

    for j = 1:p
        Sj = S(j,:)';
        Lj = (Ez * Sj) / Ezz;
        switch penalty
            case 'L2'
                Lj = Lj / (1 + lambda);
            case 'L1'
                Lj = sign(Lj) .* max(abs(Lj) - lambda, 0);
            case 'EN'
                alpha = 0.5;
                Lj = sign(Lj) .* max(abs(Lj) - lambda * alpha, 0) / (1 + lambda * (1 - alpha));
        end
        Lambda(j,:) = Lj;
    end
    Psi = diag(S - Lambda * Ez);
    Psi(Psi < 1e-4) = 1e-4;
end
end
