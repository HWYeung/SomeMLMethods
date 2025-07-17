function [L_best, F_history, F_train_history, epoch, normGrad, ProbFunct] = NCMML_v2(X, ...
    label, batchsize, regularization, margins, improve, maxepoch, percent, CVset, Linit, weightimportance)
% NCMML: Nearest Class Mean Metric Learning
%   Learns a Mahalanobis-like transformation matrix L using log-likelihood
%   score formulation and gradient ascent.
%
% Inputs:
%   - X:          n x d matrix of data
%   - label:      n x 1 vector of class labels
%   - batchsize:  number of samples per gradient step
%   - regularization: scalar for logdet regularization (initial value)
%   - margins:    rank of transformation matrix L (rows)
%   - improve:    early stopping patience
%   - maxepoch:   maximum number of epochs
%   - percent:    training proportion threshold (used in early stop)
%   - CVset:      cell array {TrainInd, ValidInd, TrainPer} (optional)
%   - Linit:      optional initial L
%   - weightimportance: class weight vector (optional)
%
% Outputs:
%   - L_best:     learned transformation matrix
%   - F_history:  validation score history
%   - F_train_history: training score history
%   - epoch:      number of epochs run
%   - normGrad:   gradient norm history
%   - ProbFunct:  function handle for transformed class probabilities

if nargin < 11
    weightimportance = [];
end

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

LabelMatrix = label == unique(label)';
if isempty(Linit) || all(all(Linit == 0))
    L = initialisedW([margins size(X,2)], 'xavier');
    L = L / norm(L);
else
    L = Linit;
end

if isempty(weightimportance)
    Wimpt = ones(1, length(unique(label)));
else
    Wimpt = weightimportance;
end

normGrad = [];
adapt_reg = regularization;
adapt_rate = 0.01;
min_reg = 1e-4;
max_reg = 10;

if isempty(TrainInd)
    ClassMean = X' * LabelMatrix ./ sum(LabelMatrix);
else
    ClassMean = X(TrainInd,:)' * LabelMatrix(TrainInd,:) ./ sum(LabelMatrix(TrainInd,:));
end

P = NCMC(X, L, ClassMean) + eps;
Weight = sum(LabelMatrix .* Wimpt ./ sum(LabelMatrix .* Wimpt, 'all'), 2);
F_val = compute_log_likelihood_score(P, LabelMatrix, Weight, ValidInd) - adapt_reg * logdet_penalty(L);
F_train = compute_log_likelihood_score(P, LabelMatrix, Weight, TrainInd);
F = F_val;
F_history = F_val;
F_train_history = F_train;
L_best = L;
ProbFunct = @(x) NCMC(x, L_best, ClassMean);
push = 1; epoch = 1; valid = 0;

figure; hold on;
title('Training and Validation Score');
xlabel('Iteration'); ylabel('Score');
legend('Train','Validation');

while epoch <= maxepoch && valid <= improve
    batches = ceil(Trainsize / batchsize);
    shuffle = randperm(Trainsize);
    for i = 1:batches
        idx = shuffle((i-1)*batchsize +1 : min(i*batchsize, Trainsize));
        if isempty(TrainInd)
            Grad = NCMMLGradient(X, ClassMean, P, LabelMatrix, Weight, idx);
        else
            Grad = NCMMLGradient(X, ClassMean, P, LabelMatrix, Weight, TrainInd(idx));
        end

        logdet_grad = 2 * ((L' * L + eps * eye(size(L,2))) \ L);
        Grad = L * Grad + adapt_reg * logdet_grad;
        normGrad = [normGrad, norm(Grad)];
        L = L + 0.1 * log(1 + push) * Grad;
        P = NCMC(X, L, ClassMean) + eps;

        F_val = compute_log_likelihood_score(P, LabelMatrix, Weight, ValidInd) - adapt_reg * logdet_penalty(L);
        F_train = compute_log_likelihood_score(P, LabelMatrix, Weight, TrainInd);

        if F_val > F
            F = F_val;
            L_best = L;
            valid = 0;
            ProbFunct = @(x) NCMC(x, L_best, ClassMean);
        else
            valid = valid + 1;
        end

        adapt_reg = min(max(adapt_reg * (1 + adapt_rate * sign(F_train - F_val)), min_reg), max_reg);

        if abs(F_val - F_history(end)) / abs(F_val) < 1e-4
            push = push + 1;
        else
            push = 1;
        end

        F_history = [F_history, F_val];
        F_train_history = [F_train_history, F_train];

        plot(length(F_history), F_train, 'b.');
        plot(length(F_history), F_val, 'r.');
        drawnow;

        if isnan(F_val) || valid > improve
            break
        end
    end
    epoch = epoch + 1;
end
end

function [Grad] = NCMMLGradient(X, ClassMean, P, LabelMatrix, Weight, Ibatch)
X_ij_all = permute(X(Ibatch,:), [3 2 1]) - ClassMean';
Alpha = P(Ibatch,:) - LabelMatrix(Ibatch,:);
Project = permute((Weight(Ibatch) .* Alpha)', [1 3 2]);
Grad = sum(pagemtimes(X_ij_all, 'transpose', Project .* X_ij_all, 'none'), 3) ./ sum(Weight(Ibatch));
end

function Probability = NCMC(X, L, ClassMean)
LX = L * X';
LClassMean = L * ClassMean;
vSsqLX = sum(LX .^ 2, 1);
Exp = -(vSsqLX' - 2 * (LX') * LClassMean + sum(LClassMean.^2, 1));
Exp = Exp - max(Exp, [], 2);
modi_exp = exp(Exp);
Probability = modi_exp ./ sum(modi_exp, 2);
end

function penalty = logdet_penalty(L)
[~, S, ~] = svd(L, 'econ');
penalty = sum(log(diag(S).^2 + eps));
end

function score = compute_log_likelihood_score(P, LabelMatrix, Weight, ValidInd)
% COMPUTE_LOG_LIKELIHOOD_SCORE
% Computes the average weighted log-probability score for the true class.
%
% Inputs:
%   P           : n x k matrix of predicted probabilities for each class.
%   LabelMatrix : n x k one-hot encoded true labels.
%   Weight      : n x 1 vector of sample weights.
%   ValidInd    : vector of indices to consider (e.g. validation set).
%
% Output:
%   score       : weighted average log-probability of correct classes.
%
if isempty(ValidInd)
    ValidInd = 1:size(P,1);
end
score = sum(Weight(ValidInd) .* sum(LabelMatrix(ValidInd,:) .* log(P(ValidInd,:)), 2)) ./ sum(Weight(ValidInd));
end

function W = initialisedW(sz, method)
if strcmp(method, 'glorot')
    W = (2 * rand(sz) - 1) * sqrt(3 / sum(sz));
elseif strcmp(method, 'xavier')
    W = randn(sz) * sqrt(1 / sum(2 * sz));
elseif strcmp(method, 'he')
    W = randn(sz) * sqrt(2 / sum(sz));
end
end