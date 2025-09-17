function [L_best, F_history, F_train_history, epoch, normGrad, ProbFunct, lambda_history] = NCMML_v2(X, ...
    label, batchsize, lambda_frob_init, lambda_spec_init, margins, improve, maxepoch, percent, CVset, Linit, weightimportance)
% NCMML_v2: Nearest Class Mean Metric Learning with dual adaptive regularization
%   Learns a Mahalanobis-like transformation matrix L using log-likelihood
%   score formulation and gradient ascent with two adaptive regularization terms:
%   1. Adaptive Frobenius norm regularization (lambda_frob) based on generalization gap
%   2. Adaptive spectral regularization (lambda_spec) based on condition number
%
% Inputs:
%   - X:          n x d matrix of data (required)
%   - label:      n x 1 vector of class labels (required)
%   - batchsize:  number of samples per gradient step (default: 64)
%   - lambda_frob_init: initial Frobenius regularization (default: 1e-3)
%   - lambda_spec_init: initial spectral regularization (default: 1e-4)
%   - margins:    rank of transformation matrix L (default: size(X,2))
%   - improve:    early stopping patience (default: 20)
%   - maxepoch:   maximum number of epochs (default: 500)
%   - percent:    training proportion threshold (default: 0.8)
%   - CVset:      cell array {TrainInd, ValidInd, TrainPer} (default: [])
%   - Linit:      optional initial L (default: Xavier initialization)
%   - weightimportance: class weight vector (default: ones)
%
% Outputs:
%   - L_best:     learned transformation matrix
%   - F_history:  validation score history
%   - F_train_history: training score history
%   - epoch:      number of epochs run
%   - normGrad:   gradient norm history
%   - ProbFunct:  function handle for transformed class probabilities
%   - lambda_history: history of regularization parameters

% Set default values for optional parameters
if nargin < 3 || isempty(batchsize)
    batchsize = 64;
end

if nargin < 4 || isempty(lambda_frob_init)
    lambda_frob_init = 1e-3;  % Good default for Frobenius regularization
end

if nargin < 5 || isempty(lambda_spec_init)
    lambda_spec_init = 1e-4;  % Good default for spectral regularization
end

if nargin < 6 || isempty(margins)
    margins = size(X, 2);     % Default to full rank
end

if nargin < 7 || isempty(improve)
    improve = 20;             % Default early stopping patience
end

if nargin < 8 || isempty(maxepoch)
    maxepoch = 500;           % Default maximum epochs
end

if nargin < 9 || isempty(percent)
    percent = 0.8;            % Default training proportion
end

if nargin < 10 || isempty(CVset)
    CVset = [];
end

if nargin < 11 || isempty(Linit)
    Linit = [];               % Will use Xavier initialization
end

if nargin < 12 || isempty(weightimportance)
    weightimportance = [];    % Will use uniform weights
end

% Rest of the function remains the same...
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
lambda_frob = lambda_frob_init;
lambda_spec = lambda_spec_init;

% Adaptive regularization parameters
frob_adapt_rate = 0.1;      % Adaptation rate for Frobenius regularization
spec_adapt_rate = 0.05;     % Adaptation rate for spectral regularization
min_frob = 1e-6;           % Minimum Frobenius regularization
max_frob = 1.0;            % Maximum Frobenius regularization
min_spec = 1e-8;           % Minimum spectral regularization
max_spec = 0.1;            % Maximum spectral regularization
target_cond = 100;         % Target condition number for stability

% Learning rate parameters
base_lr = 0.1;
lr_decay = 0.95;
lr_decay_epoch = 50;

if isempty(TrainInd)
    ClassMean = X' * LabelMatrix ./ sum(LabelMatrix);
else
    ClassMean = X(TrainInd,:)' * LabelMatrix(TrainInd,:) ./ sum(LabelMatrix(TrainInd,:));
end

P = NCMC(X, L, ClassMean) + eps;
Weight = sum(LabelMatrix .* Wimpt ./ sum(LabelMatrix .* Wimpt, 'all'), 2);

% Compute initial scores with both regularization terms
spec_penalty = lambda_spec * logdet_penalty(L);
frob_penalty = lambda_frob * norm(L, 'fro')^2;
F_val = compute_log_likelihood_score(P, LabelMatrix, Weight, ValidInd) - spec_penalty - frob_penalty;
F_train = compute_log_likelihood_score(P, LabelMatrix, Weight, TrainInd);
F = F_val;
F_history = F_val;
F_train_history = F_train;
lambda_history = [lambda_frob, lambda_spec];
L_best = L;
ProbFunct = @(x) NCMC(x, L_best, ClassMean);
push = 1; epoch = 1; valid = 0;

figure; 
subplot(2,1,1); hold on;
title('Training and Validation Score');
xlabel('Iteration'); ylabel('Score');
legend('Train','Validation');

subplot(2,1,2); hold on;
title('Regularization Parameters');
xlabel('Iteration'); ylabel('Lambda');
legend('Frobenius', 'Spectral');

while epoch <= maxepoch && valid <= improve
    % Adjust learning rate based on epoch
    current_lr = base_lr * (lr_decay ^ floor(epoch / lr_decay_epoch));
    
    batches = ceil(Trainsize / batchsize);
    shuffle = randperm(Trainsize);
    
    for i = 1:batches
        idx = shuffle((i-1)*batchsize +1 : min(i*batchsize, Trainsize));
        if isempty(TrainInd)
            Grad = NCMMLGradient(X, ClassMean, P, LabelMatrix, Weight, idx);
        else
            Grad = NCMMLGradient(X, ClassMean, P, LabelMatrix, Weight, TrainInd(idx));
        end

        % Compute both regularization gradients
        logdet_grad = 2 * lambda_spec * ((L' * L + eps * eye(size(L,2))) \ L);
        frob_grad = 2 * lambda_frob * L;
        
        % Combined gradient
        Grad = L * Grad + logdet_grad + frob_grad;
        normGrad = [normGrad, norm(Grad)];
        
        % Update L with current learning rate
        L = L + current_lr * log(1 + push) * Grad;
        P = NCMC(X, L, ClassMean) + eps;

        % Compute new scores with both regularization terms
        spec_penalty = lambda_spec * logdet_penalty(L);
        frob_penalty = lambda_frob * norm(L, 'fro')^2;
        F_val = compute_log_likelihood_score(P, LabelMatrix, Weight, ValidInd) - spec_penalty - frob_penalty;
        F_train = compute_log_likelihood_score(P, LabelMatrix, Weight, TrainInd);

        % Adaptive regularization based on condition number and generalization gap
        if ~isempty(ValidInd)
            % Calculate current condition number
            s = svd(L);
            cond_number = max(s) / (min(s) + eps);
            
            % Adjust spectral regularization based on condition number
            spec_error = (cond_number - target_cond) / target_cond;
            lambda_spec = min(max(lambda_spec * (1 + spec_adapt_rate * spec_error), min_spec), max_spec);
            
            % Adjust Frobenius regularization based on generalization gap
            generalization_gap = (F_train - F_val) / (abs(F_train) + eps);
            lambda_frob = min(max(lambda_frob * (1 + frob_adapt_rate * generalization_gap), min_frob), max_frob);
        end

        % Update best solution
        if F_val > F
            F = F_val;
            L_best = L;
            valid = 0;
            ProbFunct = @(x) NCMC(x, L_best, ClassMean);
        else
            valid = valid + 1;
        end

        % Adjust push factor based on progress
        if abs(F_val - F_history(end)) / abs(F_val) < 1e-4
            push = push + 1;
        else
            push = max(1, push * 0.9);  % Reduce push if progress is made
        end

        F_history = [F_history, F_val];
        F_train_history = [F_train_history, F_train];
        lambda_history = [lambda_history; [lambda_frob, lambda_spec]];

        subplot(2,1,1);
        plot(length(F_history), F_train, 'b.');
        plot(length(F_history), F_val, 'r.');
        
        subplot(2,1,2);
        plot(length(F_history), lambda_frob, 'g.');
        plot(length(F_history), lambda_spec, 'm.');
        
        drawnow;

        if isnan(F_val) || valid > improve
            break
        end
    end
    
    % Display progress
    if mod(epoch, 10) == 0
        fprintf('Epoch %d: F_val=%.4f, F_train=%.4f, cond=%.1f, lambda_frob=%.2e, lambda_spec=%.2e\n', ...
                epoch, F_val, F_train, cond_number, lambda_frob, lambda_spec);
    end
    
    epoch = epoch + 1;
    
    % Early stopping check
    if valid > improve
        fprintf('Early stopping at epoch %d\n', epoch);
        break;
    end
    close all;
end

fprintf('Final: lambda_frob=%.2e, lambda_spec=%.2e, F_val=%.4f, F_train=%.4f\n', ...
        lambda_frob, lambda_spec, F_val, F_train);
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
