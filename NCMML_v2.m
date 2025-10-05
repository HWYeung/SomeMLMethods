function [L_best, F_history, epoch, normGrad, ProbFunct, lambda_history] = NCMML_v2(X, ...
    label, batchsize, lambda_frob_init, lambda_spec_init, margins, improve, maxepoch, percent, CVset, Linit, weightimportance, CaseControl)
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
    lambda_spec_init = 1e-5;  % Good default for spectral regularization
end

if nargin < 6 || isempty(margins)
    margins = size(X, 2);     % Default to full rank
end

if nargin < 7 || isempty(improve)
    improve = 10000;             % Default early stopping patience
end

if nargin < 8 || isempty(maxepoch)
    maxepoch = 10;           % Default maximum epochs
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

if isempty(weightimportance)
    Wimpt = ones(1, length(unique(label)));
else
    Wimpt = weightimportance;
end

normGrad = [];
lambda_spec = lambda_spec_init;

% === DROPOUT ===
dropout_rate = 0.20; 
dropout_epoch_start = 5;

% === NESTEROV ACCELERATION
momentum = 0.85;              % Momentum coefficient
velocity = zeros([margins size(X,2)]);   % Velocity term for momentum
use_nesterov = false;

% === SIMPLIFIED REGULARIZATION - START WITH JUST NUCLEAR NORM ===
lambda_nuclear = 1e-5;           % Start very small with nuclear norm
lambda_manifold = 0;             % Disable manifold initially

% Conservative adaptation
spec_adapt_rate = 0.005;
nuclear_adapt_rate = 0.001;

% Very tight bounds
min_spec = 1e-9;
max_spec = 1e-3;
min_nuclear = 1e-7;
max_nuclear = 1e-3;

target_cond = 5;                 % More conservative target

% Much more conservative learning parameters
base_lr = 0.05;                 % 10x smaller learning rate
lr_decay = 0.99;
lr_decay_epoch = 5;
max_grad_norm = 100.0;            % Tighter gradient clipping

if isempty(TrainInd)
    ClassMean = X' * LabelMatrix ./ sum(LabelMatrix);
else
    ClassMean = X(TrainInd,:)' * LabelMatrix(TrainInd,:) ./ sum(LabelMatrix(TrainInd,:));
end

Weight = sum(LabelMatrix .* Wimpt ./ sum(LabelMatrix .* Wimpt, 'all'), 2);

if isempty(Linit) || all(all(Linit == 0))
    % Option 1: Multiple random initializations with best validation
    num_init = 100;
    best_init_score = -inf;
    best_init_L = [];
    
    for init_idx = 1:num_init
        L_candidate = initialisedW([margins size(X,2)], 'xavier');
        L_candidate = L_candidate / norm(L_candidate);
        
        % Quick validation of this initialization
        P_init = NCMC(X, L_candidate, ClassMean) + eps;
        init_score = compute_log_likelihood_score(P_init, LabelMatrix, Weight, ValidInd);
        
        if init_score > best_init_score
            best_init_score = init_score;
            best_init_L = L_candidate;
        end
    end
    L = best_init_L;
    fprintf('Selected best initialization with score: %.4f\n', best_init_score);
else
    L = Linit;
end

P = NCMC(X, L, ClassMean) + eps;


% Compute initial scores - JUST spectral + nuclear
spec_penalty = lambda_spec * logdet_penalty(L);
nuclear_penalty = lambda_nuclear * sum(svd(L));

F_val = compute_log_likelihood_score(P, LabelMatrix, Weight, ValidInd) - ...
        spec_penalty - nuclear_penalty;
F_train = compute_log_likelihood_score(P, LabelMatrix, Weight, TrainInd);
F = F_val;
F_history = [F_train; F_val];
lambda_history = [lambda_spec; lambda_nuclear];
L_best = L;
ProbFunct = @(x) NCMC(x, L_best, ClassMean);
push = 1; epoch = 1; valid = 0;

% === GRADIENT NORM TRACKING FOR ADAPTIVE CLIPPING ===
grad_norm_history = [];

while epoch <= maxepoch && valid <= improve
    current_lr = base_lr * (lr_decay ^ floor(epoch / lr_decay_epoch)); 

    if epoch >= dropout_epoch_start
        % Create dropout mask for features
        dropout_mask = (rand(1, size(X, 2)) > dropout_rate);
        X_dropout = X .* dropout_mask;
        fprintf('Applying dropout: %.1f%% features kept\n', mean(dropout_mask) * 100);
    else
        X_dropout = X;  % No dropout during warmup
    end
    
    % Update ClassMean with dropped-out features if needed
    if epoch >= dropout_epoch_start
        if isempty(TrainInd)
            ClassMean_current = X_dropout' * LabelMatrix ./ sum(LabelMatrix);
        else
            ClassMean_current = X_dropout(TrainInd,:)' * LabelMatrix(TrainInd,:) ./ sum(LabelMatrix(TrainInd,:));
        end
    else
        ClassMean_current = ClassMean;  % Use original during warmup
    end
    
    batches = ceil(Trainsize / batchsize);
    shuffle = randperm(Trainsize);
    
    for i = 1:batches
        idx = shuffle((i-1)*batchsize +1 : min(i*batchsize, Trainsize));
        if isempty(TrainInd)
            Grad = NCMMLGradient(X_dropout, ClassMean_current, P, LabelMatrix, Weight, idx);
        else
            Grad = NCMMLGradient(X_dropout, ClassMean_current, P, LabelMatrix, Weight, TrainInd(idx));
        end

        % === AGGRESSIVE GRADIENT CLIPPING AND CHECKING ===
        grad_norm = norm(Grad(:));
        grad_norm_history = [grad_norm_history, grad_norm];
        
        % Adaptive gradient clipping based on history
        if length(grad_norm_history) > 10
            median_grad_norm = median(grad_norm_history(end-9:end));
            current_max_grad = min(max_grad_norm, 2 * median_grad_norm);
        else
            current_max_grad = max_grad_norm;
        end
        
        if grad_norm > current_max_grad
            Grad = Grad * (current_max_grad / grad_norm);
            fprintf('Gradient clipped: %.2e -> %.2e\n', grad_norm, current_max_grad);
        end
        
        if any(isnan(Grad(:))) || any(isinf(Grad(:)))
            warning('NaN/Inf in gradient - skipping update');
            continue;
        end

        % === SIMPLIFIED REGULARIZATION GRADIENTS ===
        % Spectral regularization gradient (more stable computation)
        [U, S, V] = svd(L' * L + 1e-6 * eye(size(L, 2)));  % Larger epsilon
        s = diag(S);
        s_inv = s ./ (s.^2 + 1e-10);  % More stable inversion
        M_inv = V * diag(s_inv) * U';
        logdet_grad = 2 * lambda_spec * (L * M_inv);

        % Nuclear norm gradient (with safety)
        if lambda_nuclear > 0
            [U_nuc, S_nuc, V_nuc] = svd(L, 'econ');
            min_sv = min(diag(S_nuc));
            if min_sv > 1e-8  % Only apply if well-conditioned
                nuclear_grad = lambda_nuclear * U_nuc * V_nuc';
            else
                nuclear_grad = 0;
                lambda_nuclear = max(lambda_nuclear * 0.5, min_nuclear);
            end
        else
            nuclear_grad = 0;
        end

        % Combined gradient (much simpler)
        Grad = L * Grad + logdet_grad + nuclear_grad;
        total_grad_norm = norm(Grad(:));
        normGrad = [normGrad, total_grad_norm];
        
        % Check update magnitude
        update_step = current_lr * min(log(1 + push), 2.0) * Grad;
        update_norm = norm(update_step(:));
        if update_norm > 0.1 * norm(L(:))
            update_step = update_step * (0.1 * norm(L(:)) / update_norm);
            fprintf('Update step too large: %.2e, scaling down\n', update_norm);
        end
        
        if use_nesterov
            % === EFFICIENT NESTEROV APPROXIMATION ===
            % Store current velocity
            velocity_prev = velocity;
            
            % Update velocity with current gradient
            velocity = momentum * velocity - update_step;
            
            % Nesterov update: jump ahead then correct
            L = L - momentum * velocity_prev + (1 + momentum) * velocity;
            if valid == 0  % Making progress
                momentum = min(0.9, momentum + 0.01);
            else
                momentum = max(0.2, momentum * 0.95);
            end
        else
            % Standard update
            L = L + update_step;
        end
        
        % === STRICT NUMERICAL STABILITY CHECK ===
        if any(isnan(L(:))) || any(isinf(L(:))) || norm(L(:)) > 1e6
            warning('Numerical instability detected - resetting to best');
            L = L_best;
            current_lr = current_lr * 0.1;
            continue;
        end
        
        P = NCMC(X, L, ClassMean) + eps;

        % Compute new scores
        spec_penalty = lambda_spec * logdet_penalty(L);
        nuclear_penalty = lambda_nuclear * sum(svd(L));
        
        % F_val_new = compute_log_likelihood_score(P, LabelMatrix, Weight, ValidInd);
        % F_train_new = compute_log_likelihood_score(P, LabelMatrix, Weight, TrainInd);

        F_val_new = compute_accuracy(P, CaseControl, ValidInd);
        F_train_new = compute_accuracy(P, CaseControl, TrainInd);
        
        % === EXTREME SAFETY CHECK ===
        if abs(F_val_new) > 1e3 || isnan(F_val_new) || isinf(F_val_new)
            warning('Validation loss suspicious: %.2e - reverting', F_val_new);
            L = L - update_step;  % Revert update
            current_lr = current_lr * 0.2;
            
            % Reduce regularization if causing issues
            lambda_nuclear = lambda_nuclear * 0.5;
            lambda_spec = lambda_spec * 0.5;
            
            continue;
        end
        
        F_val = F_val_new;
        F_train = F_train_new;

        % === VERY CONSERVATIVE ADAPTATION ===
        if ~isempty(ValidInd) && F_val > -1e3  % Only adapt if reasonable
            s = svd(L);
            cond_number = max(s) / (min(s) + eps);
            
            if cond_number > 10  % Only adjust if really needed
                spec_error = (cond_number - target_cond) / target_cond;
                lambda_spec = min(max(lambda_spec * (1 + 0.001 * sign(spec_error)), min_spec), max_spec);
            end
            generalization_gap = (F_train - F_val) / (abs(F_train) + eps);
            lambda_nuclear = min(max(lambda_nuclear * (1 + 0.0005 * generalization_gap), min_nuclear), max_nuclear);
        end

        % Update best solution
        if F_val > F && abs(F_val) < 1e3 && F_train >= F_val % Only update if reasonable
            F = F_val;
            L_best = L;
            valid = 0;
            ProbFunct = @(x) NCMC(x, L_best, ClassMean);
        else
            valid = valid + 1;
        end

        % Adjust push factor very conservatively
        if valid == 0 && abs(F_val - F_history(2,end)) / abs(F_val) < 1e-3
            push = min(push + 0.1, 3.0);
        else
            push = max(1, push * 0.95);
        end

        F_history = [F_history [F_train; F_val]];
        lambda_history = [lambda_history [lambda_spec; lambda_nuclear]];

        % Simple plotting
        if mod(i, 5) == 0
            subplot(2,1,1);
            plot(1:length(F_history), F_history(1,:), 'b-', 1:length(F_history), F_history(2,:), 'r-');
            xlabel('Iteration'); ylabel('Loss'); title('Loss');
            legend('Train', 'Val');
            
            subplot(2,1,2);
            semilogy(1:length(lambda_history), lambda_history(1,:), 'g-', ...
                     1:length(lambda_history), lambda_history(2,:), 'm-');
            xlabel('Iteration'); ylabel('Lambda'); title('Regularization');
            legend('Spectral', 'Nuclear');
            drawnow;
        end

        if valid > improve
            break
        end
    end
    
    % Minimal progress reporting
    if mod(epoch, 1) == 0
        s = svd(L);
        cond_number = max(s) / (min(s) + eps);
        fprintf('Epoch %d: F_val=%.3f%%, F_train=%.3f%%, cond=%.1f, grad_norm=%.2e\n', ...
                epoch, F_val, F_train, cond_number, median(grad_norm_history(max(1,end-9):end)));
    end
    
    epoch = epoch + 1;
    
    if valid > improve || current_lr < 1e-8
        fprintf('Stopping: valid=%d, lr=%.2e\n', valid, current_lr);
        break;
    end
end

fprintf('Final: F_val=%.3f%%, F_train=%.4f\n', F_val, F_train);
fprintf('Final Lambdas: spec=%.2e, nuclear=%.2e\n', lambda_spec, lambda_nuclear);
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

function accuracy = compute_accuracy(P, CaseControl, ValidInd)
    % Compute case-control accuracy
    % P: probability matrix (n_samples x n_classes)
    % LabelMatrix: one-hot encoded labels
    % CaseControl: binary vector (0=control, 1=case)
    % ValidInd: validation indices
    
    if isempty(ValidInd)
        ValidInd = 1:size(P,1);
    end
    
    % Get predicted clusters (MATLAB 1-based indexing)
    [~, pred_clusters] = max(P(ValidInd, :), [], 2);
    pred_clusters = pred_clusters - 1;  % Convert to 0-based: 1→0, 2→1, 3→2, etc.
    
    % True case/control status
    true_status = CaseControl(ValidInd);
    
    % Predicted status: 0 if cluster 0, 1 if cluster > 0
    pred_status = (pred_clusters > 0);
    
    % Case-control accuracy
    accuracy = mean(pred_status == true_status) * 100;
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
