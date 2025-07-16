function [L_best, F2, k, Grad, Exp] = MahalanobisLearning(X, label, batchsize, improve, maxepoch, percent, CVset, boundary)
% MahalanobisLearning Learns a Mahalanobis distance metric via focal loss optimization
%
% This function learns a linear transformation matrix L such that the 
% Mahalanobis distance induced by M = L'*L respects class similarity 
% constraints, optimized with a focal loss to handle hard pairs and 
% an adaptive batch training scheme. The method includes a thresholding 
% margin parameter `boundary` to control intra- and inter-class pair 
% distance separation.
%
% Inputs:
%   X          - (N x D) data matrix with N samples and D features
%   label      - (N x 1) vector of integer class labels
%   batchsize  - scalar specifying mini-batch size for training
%   improve   - scalar, number of epochs allowed without improvement before stopping
%   maxepoch   - scalar, maximum number of epochs to run training
%   percent    - scalar, target performance metric threshold for early stopping
%   CVset      - cell array {TrainInd, ValidInd, TrainPer} for cross-validation indices
%                or empty to use all data for training
%   boundary   - scalar margin parameter to scale positive vs negative pair distance thresholds
%
% Outputs:
%   L_best     - (r x D) learned linear transformation matrix (r ≤ D)
%   F2         - vector of focal loss values recorded during training
%   k          - scalar, number of iterations performed
%   Grad       - gradient matrix computed at last iteration
%   Exp        - pairwise squared Mahalanobis distances at last iteration
%
% Description:
%   The algorithm learns a low-rank matrix L mapping original data 
%   into a new space where distances reflect label similarity. 
%   It uses a focal loss-based weighting scheme to emphasize hard 
%   training pairs and adaptively adjusts learning via mini-batches. 
%
%   The `boundary` parameter controls the margin between intra-class 
%   and inter-class distances by scaling the threshold applied to 
%   distances during the indicator mask calculation, effectively tuning 
%   how strictly positives and negatives are separated.
%
%   Training stops early if performance fails to improve for `improve` 
%   epochs or reaches `percent` target.
%
% Notes:
%   - Initial matrix L is initialized via Xavier method and normalized.
%   - The function assumes input labels are categorical or integer classes.
%   - Outputs include diagnostic tracking for loss and largest eigenvalue 
%     of M = L*L', useful for monitoring convergence and conditioning.
%
% Example usage:
%   [L,loss_hist,iters,grad,distances] = MahalanobisLearning(X,labels,32,10,100,0.95,[],2);
%
% See also: MahalanobisGradient, initialisedW

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

LabelMatrix = label == label';
L = initialisedW([min(floor(size(X,1)), floor(size(X,2)*2/3)) size(X,2)], 'xavier');
L = L / sqrt(trace(L.' * L));

LX = L * X.';
vSsqLX = sum(LX .^ 2, 1);
Exp = vSsqLX.' + vSsqLX - (2 * (LX.' * LX));

distthres = mean(Exp, "all");
Distthres = zeros(size(Exp));
Distthres(LabelMatrix == 1) = (1 / boundary) * distthres;
Distthres(LabelMatrix == 0) = boundary * distthres;

Indicator = (Exp > Distthres) .* LabelMatrix + (Exp < Distthres) .* (1 - LabelMatrix);

modi_exp = max(1e-323, exp(-Exp));
modi_exp(logical(eye(size(modi_exp)))) = 0;
P = modi_exp ./ (sum(modi_exp));
P = P - diag(diag(P));

C = P .* LabelMatrix;
P_i = sum(C);

if isempty(ValidInd)
    F1 = mean(P_i);
else
    F1 = mean(P_i(ValidInd));
end

k = 1;
F2 = F1;
Eig = eigs(L * L.', 1);
valid = 0;
F = F1;
L_best = L;

while k < maxepoch && valid <= improve
    batches = ceil(Trainsize / batchsize);
    shuffle = randperm(Trainsize);
    
    for i = 1:batches
        idx = shuffle((i - 1) * batchsize + 1 : min(i * batchsize, Trainsize));
        shuffle2 = randperm(Trainsize);
        
        for j = 1:batches
            idx2 = shuffle2((j - 1) * batchsize + 1 : min(j * batchsize, Trainsize));
            
            if isempty(TrainInd)
                Grad = MahalanobisGradient(X, P, P_i, C, idx, idx2, Indicator);
            else
                Grad = MahalanobisGradient(X, P, P_i, C, TrainInd(idx), TrainInd(idx2), Indicator);
            end
            
            % Hyperparameters
            lambda_reg = 0.01;      % Regularisation weight for log-det
            epsilon = 1e-4;         % Stabilisation for matrix inversion
            base_lr = 0.01;         % Base learning rate
            scale_factor = 1 / sqrt(length(idx) * length(idx2));  % Batch-size based scaling
            
            % Log-det regularisation, spectral regularisation
            LLT = L * L.' + epsilon * eye(size(L,1));
            logdet_grad = -2 * (LLT \ L);  % Gradient of -logdet(LLT)
            
            % Combine gradients
            Grad_total = Grad + lambda_reg * logdet_grad;
            
            % Compute final gradient
            Grad_total = 2 * L * Grad_total;
            
            % Optional: gradient norm scaling (adaptive step)
            grad_norm = norm(Grad_total, 'fro');
            if grad_norm > 0
                adaptive_lr = base_lr * scale_factor / grad_norm;
            else
                adaptive_lr = base_lr * scale_factor;
            end
            
            % Update L
            L = L + adaptive_lr * Grad_total;
            % L = L / sqrt(trace(L.' * L));  % optional renormalisation

            LX = L * X.';
            vSsqLX = sum(LX .^ 2, 1);
            Exp = vSsqLX.' + vSsqLX - (2 * (LX.' * LX));

            Indicator = (Exp > Distthres) .* LabelMatrix + (Exp < Distthres) .* (1 - LabelMatrix);
            modi_exp = max(1e-323, exp(-Exp));
            modi_exp(logical(eye(size(modi_exp)))) = 0;
            P = modi_exp ./ (sum(modi_exp));
            C = P .* LabelMatrix;
            P_i = sum(C);
            
            if isempty(ValidInd)
                F1 = mean(P_i);
                FT = mean(P_i);
            else
                F1 = mean(P_i(ValidInd));
                FT = mean(P_i(TrainInd));
            end
            
            if F < F1
                F = F1;
                valid = 0;
                L_best = L;
            else
                valid = valid + 1;
            end
            
            if isnan(F1) || F1 > percent || FT > TrainPer || valid > improve
                F2 = [F2 F1];
                break
            end
            
            F2 = [F2 F1];
            Eig = [Eig eigs(L * L.', 1)];
            
            subplot(2,1,1)
            plot(1:length(F2)-1, F2(2:end));
            subplot(2,1,2)
            plot(1:length(Eig)-1, Eig(2:end));
            drawnow;
            
            k = k + 1;
        end
        if isnan(F1) || F1 > percent || FT > TrainPer || valid > improve
            break
        end
    end
    if isnan(F1) || F1 > percent || FT > TrainPer || valid > improve
        break
    end
end

end

function [Grad] = MahalanobisGradient(X, P, P_i, C, Ibatch, Jbatch, Indicator)

X_ij_all = permute(X(Ibatch,:), [3 2 1]) - X(Jbatch,:);
focal1 = (P(Ibatch, Jbatch)' - (C(Ibatch, Jbatch)' ./ P_i(Ibatch))) .* ((1 - P_i(Ibatch)).^2);
focal2 = ((P_i(Ibatch) .* P(Ibatch, Jbatch)') - (C(Ibatch, Jbatch)')) .* (-2 * (1 - P_i(Ibatch))) .* log(P_i(Ibatch));
Project = permute(focal1 + focal2, [1 3 2]);

Grad = sum(pagemtimes(X_ij_all, 'transpose', Project .* X_ij_all, 'none'), 3);

end

function W = initialisedW(sz, method)
% Glorot/Xavier/He initialization

if strcmp(method, 'glorot')
    W = (2 * rand(sz) - 1) * sqrt(3 / sum(sz));
elseif strcmp(method, 'xavier')
    W = randn(sz) * sqrt(1 / sum(2 * sz));
elseif strcmp(method, 'he')
    W = randn(sz) * sqrt(2 / sum(sz));
end

end
