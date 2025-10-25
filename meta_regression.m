function [model_loss, additive_feature_importance] = meta_regression(X, Y, train_validation_test_ind, ...
    regression_type, child_model_num)
    %META_REGRESSION Train a two-level meta-regression model with bootstrapped child learners
    %
    %  This function implements a meta-regression framework that trains multiple
    %  "child" models (each on a bootstrapped sample and random feature subset),
    %  then fits a top-level ("meta") model on their predictions.
    %
    %  The procedure performs automatic lambda selection based on validation loss
    %  while enforcing a "no undertraining" rule (train_loss < valid_loss).
    %
    %  Inputs:
    %    X : [nSamples x nFeatures] feature matrix
    %        Each row is an observation, each column a feature.
    %
    %    Y : [nSamples x 1] target vector
    %
    %    train_validation_test_ind : {train_idx, valid_idx, test_idx}
    %        Cell array containing sample indices for each split.
    %        If omitted, all samples are used for all splits.
    %
    %    regression_type : "linear" or "classification"
    %        Type of model to train for both child and meta levels.
    %
    %    child_model_num : integer
    %        Number of child models to train (default = 20).
    %
    %  Outputs:
    %    model_loss : [3 x 1] vector of losses
    %        [train_loss; valid_loss; test_loss] at the selected lambda.
    %
    %    additive_feature_importance : [nFeatures x 1] vector
    %        Sum of all child-level feature coefficients across all models.
    %
    %  Example:
    %    lambdas = logspace(-6,1,300);
    %    [loss, importance] = meta_regression(X, y, splits, "linear", 10);
    %
    %  See also: fitrlinear, fitclinear, normalize

    % --- Handle optional arguments ---
    if nargin < 3
        train_ind = 1:size(X, 1);
        valid_ind = 1:size(X, 1);
        test_ind  = 1:size(X, 1);
    else
        train_ind = train_validation_test_ind{1};
        valid_ind = train_validation_test_ind{2};
        test_ind  = train_validation_test_ind{3};
    end

    if nargin < 4
        regression_type = "linear";
    end

    if nargin < 5
        child_model_num = 20;
    end

    % --- Regularization path ---
    lambdas = logspace(-6, 1, 300);

    % --- Normalize training data (use same params for val/test) ---
    [~, centers, scales] = normalize(X(train_ind, :));
    X = normalize(X, "center", centers, "scale", scales);

    % --- Define model and loss functions ---
    model_and_loss_funct = cell(2, 1);
    switch regression_type
        case "linear"
            model_and_loss_funct{1} = @(x, y) fitrlinear(x, y, ...
                'Learner', 'leastsquares', 'Lambda', lambdas, 'Regularization', 'ridge');
            model_and_loss_funct{2} = @(yhat, ytrue) mean((yhat - ytrue).^2);

        case "classification"
            model_and_loss_funct{1} = @(x, y) fitclinear(x, y, ...
                'Learner', 'logistic', 'Lambda', lambdas, 'Regularization', 'ridge');
            model_and_loss_funct{2} = @(yhat, ytrue) mean(yhat ~= ytrue);

        otherwise
            error('Unknown regression_type: must be "linear" or "classification".');
    end

    % --- Train bootstrapped child models ---
    [child_feature_betas, child_bias, centers, scales] = meta_regression_children( ...
        X(train_ind, :), Y(train_ind), model_and_loss_funct, child_model_num);

    % --- Construct meta-level features from all child models ---
    meta_features = normalize(X * child_feature_betas + ...
        ones(size(X, 1), 1) * child_bias, "center", centers, "scale", scales);

    % --- Train meta model ---
    model_funct = model_and_loss_funct{1};
    loss_funct  = model_and_loss_funct{2};
    model = model_funct(meta_features(train_ind, :), Y(train_ind));

    predict_all = predict(model, meta_features);

    train_loss = loss_funct(predict_all(train_ind, :), Y(train_ind));
    valid_loss = loss_funct(predict_all(valid_ind, :), Y(valid_ind));
    test_loss  = loss_funct(predict_all(test_ind, :),  Y(test_ind));

    % --- Enforce undertraining rule ---
    valid_loss(valid_loss < train_loss) = inf;
    [~, min_valid_loss] = min(valid_loss);

    model_loss = [train_loss(min_valid_loss);
                  valid_loss(min_valid_loss);
                  test_loss(min_valid_loss)];

    % --- Aggregate additive feature importance (sum over children) ---
    additive_feature_importance = sum(child_feature_betas, 2);

end

function [child_feature_betas, child_bias, centers, scales] = meta_regression_children(x, y, ...
    model_and_loss_funct, child_model_num)
    %META_REGRESSION_CHILDREN Train bootstrapped child models for meta-regression
    %
    %  This function trains multiple "child" regression or classification models,
    %  each on a bootstrap sample and a random subset of features. It returns the
    %  learned coefficients, intercepts, and normalization parameters.
    %
    %  Inputs:
    %    x : [nSamples x nFeatures] feature matrix (rows = observations)
    %    y : [nSamples x 1] target vector
    %    model_and_loss_funct : {model_funct, loss_funct}
    %        - model_funct : @(X, Y) -> model with fields Beta [nFeat_sub x nLambda], 
    %                       Bias [1 x nLambda], PredictFcn
    %        - loss_funct  : @(yhat, ytrue) -> scalar loss
    %    child_model_num : number of child models to train
    %
    %  Outputs:
    %    child_feature_betas : [nFeatures x child_model_num] coefficients for each child
    %    child_bias          : [1 x child_model_num] intercepts for each child
    %    centers, scales     : normalization parameters of combined child outputs
    %
    %  Notes:
    %    - Each child uses a random feature subset; the size decreases with the
    %      number of children (inverse square root relationship, min 20% of features).
    %    - Validation loss is used to select the best lambda for each child,
    %      with an undertraining check (train_loss < valid_loss).

    n_samples  = size(x, 1);
    n_features = size(x, 2);

    model_funct = model_and_loss_funct{1};
    loss_funct  = model_and_loss_funct{2};

    % --- Feature ratio decreases with number of children ---
    min_ratio = 0.4;
    feature_ratio_relation = log10(child_model_num) + 1;
    feat_ratio = max(1 / feature_ratio_relation, min_ratio);

    % --- Preallocate outputs ---
    child_feature_betas = zeros(n_features, child_model_num);
    child_bias = zeros(1, child_model_num);

    % --- Train each child ---
    for child = 1:child_model_num

        % --- Bootstrap samples ---
        train_idx = randsample(n_samples, floor(0.8 * n_samples), true);
        valid_idx = setdiff(1:n_samples, unique(train_idx));

        % fallback if validation is empty
        if isempty(valid_idx)
            valid_idx = randsample(n_samples, round(0.2 * n_samples));
        end

        % --- Random feature subset ---
        n_feat_sub = max(1, round(feat_ratio * n_features));
        feat_idx = randperm(n_features, n_feat_sub);

        x_train = x(train_idx, feat_idx);
        y_train = y(train_idx);
        x_valid = x(valid_idx, feat_idx);
        y_valid = y(valid_idx);

        % --- Fit model and pick best lambda ---
        model = model_funct(x_train, y_train);

        predict_train = predict(model, x_train);
        predict_valid = predict(model, x_valid);

        train_loss = loss_funct(predict_train, y_train);
        valid_loss = loss_funct(predict_valid, y_valid);

        valid_loss(valid_loss < train_loss) = inf;
        [~, min_valid_loss] = min(valid_loss);

        best_betas = model.Beta(:, min_valid_loss);
        best_bias  = model.Bias(min_valid_loss);

        % --- Store in full feature space ---
        child_feature_betas(feat_idx, child) = best_betas;
        child_bias(child) = best_bias;
    end

    % --- Compute normalization parameters for child outputs ---
    y_heads = x * child_feature_betas + ones(n_samples, 1) * child_bias;
    [~, centers, scales] = normalize(y_heads);

end


