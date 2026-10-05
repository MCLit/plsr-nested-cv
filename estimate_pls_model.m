% ESTIMATE_PLS_MODEL
%
% PURPOSE
%   Main analysis script for the FU2 PLS analysis. Fits five
%   Partial Least Squares (PLS) regression models that each predict a set
%   of 9 mental-health outcomes from a different combination of predictors
%   (childhood trauma, cognition, or both), using nested cross-validation
%   to choose the number of PLS components for each model. Also prints a
%   summary of loadings and VIP (Variable Importance in Projection) scores
%   for every model.
%
% PIPELINE POSITION
%   This is the first script to run. It produces FINAL_MODELS.MAT, which
%   is required as input by:
%       - results.m
%
% INPUT FILE (must be in the current working directory)
%   CTQ_DAWBA_CANTAB_FU2_complete.csv
%       Subject-level table containing:
%         - FU2_age, sex, recruitment.centre  (covariates / site)
%         - CTQ_* columns                     (Childhood Trauma Questionnaire subscales)
%         - AGN_*, CGT_*, RPV_*, SWM_*         (CANTAB cognitive task measures)
%         - FU2_<disorder> columns             (mental-health outcome scores, see Y_labels)
%
% PIPELINE
%   This single script performs:
%       1. Data preparation
%       2. Nested CV component selection
%       3. Final whole-sample PLS fitting
%       4. VIP calculation
%       5. Whole-sample predictions and performance statistics
%
% OUTPUT FILE
%   final_models.mat
%       Contains:
%         all_models_combined  - 1x5 cell array, one struct per model (see run_model below)
%         Y_labels             - names of the 9 outcome variables
%         model_names          - human-readable names of the 5 predictor sets
%
% MODELS FITTED (all_models_combined{1..5})
%   1) Trauma           - age, sex + 5 CTQ abuse/neglect subscales
%   2) Cognition         - age, sex + 11 CANTAB cognitive measures
%   3) Trauma+Cog        - age, sex + trauma subscales + cognitive measures
%   4) Threat/Dep         - age, sex + CTQ threat & deprivation dimensions
%   5) Threat/Dep+Cog     - age, sex + threat/deprivation + cognitive measures
%
%       Each model contains:
%         - whole_data
%         - cv_results
%         - final_ncomps
%         - x_loadings
%         - y_loadings
%         - x_scores
%         - y_scores
%         - beta
%         - MSE
%         - PCTVAR
%         - stats
%         - vip_scores
%         - actuals
%         - predictions
%         - performance
%
% DEPENDENCIES
%   MATLAB Statistics and Machine Learning Toolbox

clear
clc

data = readtable('CTQ_DAWBA_CANTAB_FU2_complete.csv', 'VariableNamingRule', 'preserve');

%% Extract variables
% Outcome variables: 9 FU2 mental-health/behavioural measures
Y_labels = {'FU2_specphobia', 'FU2_socphobia', 'FU2_agoraphobia', 'FU2_ocd', ...
    'FU2_anxiety', 'FU2_depression', 'FU2_ptsd','FU2_conductdisorder', 'FU2_hyperactivity'};
y = table2array(data(:, Y_labels));

% Covariates: age, sex (coded 1 = male, 0 = female/other), and recruitment
% site, recoded to consecutive integers 1..n_sites via unique().
age_data = data.FU2_age;
sex_numeric = strcmp(data.sex, 'M');
[~, ~, site_data] = unique(data.("recruitment.centre"));

%% Define predictors
% Each X_* matrix is [age, sex, <predictor block>]. Age and sex are
% included as covariates in every model so that trauma/cognition effects
% are estimated over and above basic demographics.

% Model 1: childhood trauma subscales (CTQ abuse/neglect dimensions)
X_trauma = [age_data, sex_numeric, table2array(data(:, {'CTQ_emotional_abuse', 'CTQ_physical_abuse', 'CTQ_sexual_abuse', 'CTQ_emotional_neglect', 'CTQ_physical_neglect'}))];

% Model 2: cognitive task performance (CANTAB measures)
X_cognition = [age_data, sex_numeric, table2array(data(:, {'AGN_latency_pos', 'AGN_latency_neg', 'CGT_delay_aversion', 'CGT_deliberation', 'CGT_bet', 'CGT_quality', 'CGT_risk_adj', 'CGT_risk_taking', 'RPV_attention', 'SWM_errors', 'SWM_strategy'}))];

% Model 3: trauma subscales + cognition combined
X_combined = [age_data, sex_numeric, table2array(data(:, {'CTQ_emotional_abuse', 'CTQ_physical_abuse', 'CTQ_sexual_abuse', 'CTQ_emotional_neglect', 'CTQ_physical_neglect', 'AGN_latency_pos', 'AGN_latency_neg', 'CGT_delay_aversion', 'CGT_deliberation', 'CGT_bet', 'CGT_quality', 'CGT_risk_adj', 'CGT_risk_taking', 'RPV_attention', 'SWM_errors', 'SWM_strategy'}))];

% Model 4: threat/deprivation dimensional model of adversity (alternative
% to the 5-subscale CTQ breakdown used in Model 1)
X_threat_dep = [age_data, sex_numeric, table2array(data(:, {'CTQ_threat', 'CTQ_deprivation'}))];

% Model 5: threat/deprivation + cognition combined
X_threat_dep_cog = [age_data, sex_numeric, table2array(data(:, {'CTQ_threat', 'CTQ_deprivation', 'AGN_latency_pos', 'AGN_latency_neg', 'CGT_delay_aversion', 'CGT_deliberation', 'CGT_bet', 'CGT_quality', 'CGT_risk_adj', 'CGT_risk_taking', 'RPV_attention', 'SWM_errors', 'SWM_strategy'}))];

%% Run models
% run_model() (defined below) performs: nested CV to pick the number of
% PLS components, site-effect removal, z-scoring, and a final PLS fit on
% the whole (non-held-out) sample using the chosen number of components.
all_models_combined{1} = run_model(X_trauma, y, site_data);
all_models_combined{2} = run_model(X_cognition, y, site_data);
all_models_combined{3} = run_model(X_combined, y, site_data);
all_models_combined{4} = run_model(X_threat_dep, y, site_data);
all_models_combined{5} = run_model(X_threat_dep_cog, y, site_data);
model_names = {'Trauma', 'Cognition', 'Trauma+Cog', 'Threat/Dep', 'Threat/Dep+Cog'};

save('final_models.mat', 'all_models_combined', 'Y_labels', 'model_names');

%%
function mdl = run_model(X, y, site_data)
    % RUN_MODEL  Fit one PLS model: nested-CV component selection + final fit.
    %
    %   mdl = run_model(X, y, site_data)
    %
    %   INPUTS
    %     X          - [n_subjects x n_predictors] predictor matrix (includes
    %                  age/sex covariates plus the predictor block of interest)
    %     y          - [n_subjects x n_outcomes] outcome matrix (9 mental-health
    %                  outcome scores)
    %     site_data  - [n_subjects x 1] integer recruitment-site labels, used
    %                  to regress out site effects before PLS fitting
    %
    %   PROCESSING STEPS
    %     1. Drop rows with any NaN in X, y, or site_data (listwise deletion).
    %     2. Run pls_multivariate_nestedCV (100 repeats of nested CV) to
    %        estimate, per outer fold, the optimal number of PLS components;
    %        the final component count is the round(mean) across all folds
    %        and repeats.
    %     3. Regress out site effects from X and y (regress_site_effects),
    %        then z-score the residuals.
    %     4. Fit a final PLS model (plsregress) on the whole, site-corrected,
    %        z-scored sample using the chosen number of components.
    %     5. Compute VIP (Variable Importance in Projection) scores for each
    %        predictor.
%     6. Compute in-sample predictions and R²/Pearson's R/Spearman's rho
%        per outcome
%        (this is a whole-sample fit statistic, NOT a held-out/CV metric;
    %        cross-validated performance is stored separately in mdl.cv_results).
    %
    %   OUTPUT (mdl struct)
    %     mdl.whole_data          - X, y, site_data after NaN removal
    %     mdl.cv_results          - output of pls_multivariate_nestedCV (100 repeats)
    %     mdl.final_ncomps        - number of PLS components used in the final model
    %     mdl.x_loadings/y_loadings/x_scores/y_scores/MSE/pctvar/stats
    %                              - outputs of plsregress on the whole sample
    %     mdl.vip_scores           - VIP score per predictor
    %     mdl.actuals              - normalized X/y used for the final fit
    %     mdl.predictions          - in-sample predicted y (z-scored units)
    %     mdl.performance          - R2_per_response, R_per_response, R2_overall, R_overall

    complete_rows = ~any(isnan([X, y, site_data]), 2);
    X = X(complete_rows, :);
    y = y(complete_rows, :);
    site_data = site_data(complete_rows);

    mdl.whole_data.X = X;
    mdl.whole_data.y = y;
    mdl.whole_data.site_data = site_data;

    % Nested cross-validation: estimates out-of-sample performance and,
    % for each of the 100 repeats x 5 outer folds, the number of PLS
    % components selected by the inner CV loop.
    mdl.cv_results = pls_multivariate_nestedCV(X, y, site_data, 100);
    
    % Final number of components = round(mean) of the per-fold optimal
    % component counts across all 100 repeats x 5 outer folds.
    all_comps = [];
    for repeat = 1:100
        all_comps = [all_comps; mdl.cv_results(repeat).optimal_components];
    end
    mdl.final_ncomps = round(mean(all_comps(:)));
    
    % Remove site effects, then standardize (z-score) predictors and outcomes.
    X_regressed = regress_site_effects(X, site_data);
    y_regressed = regress_site_effects(y, site_data);
    X_norm = zscore(X_regressed);
    y_norm = zscore(y_regressed);
    
    % Final PLS fit on the whole (site-corrected, standardized) sample.
    [x_loadings, y_loadings, x_scores, y_scores, beta, PCTVAR, MSE, stats] = ...
        plsregress(X_norm, y_norm, mdl.final_ncomps);

    mdl.x_loadings = x_loadings;
    mdl.y_loadings = y_loadings;
    mdl.x_scores = x_scores;
    mdl.y_scores = y_scores;
    mdl.beta = beta;              
    mdl.MSE = MSE;
    mdl.PCTVAR = PCTVAR;          
    mdl.stats = stats;

    % VIP (Variable Importance in Projection) scores: summarize each
    % predictor's overall contribution across all retained PLS components,
    % weighted by the amount of X and Y variance each component explains.
    % W0 = component weights, normalized to unit length per component.
    W0 = stats.W ./ sqrt(sum(stats.W.^2,1));
    p = size(x_loadings,1);                                  % number of predictors
    sumSq = sum(x_scores.^2,1).*sum(y_loadings.^2,1);        % variance explained per component
    vip_scores = sqrt(p * sum(sumSq.*(W0.^2),2) ./ sum(sumSq,2));

    mdl.vip_scores = vip_scores;

    % In-sample predictions (whole-sample fit, not cross-validated) and
    % per-outcome R² / Pearson's R.
    pred = [ones(size(X_norm,1),1), X_norm] * beta;
    mdl.actuals.X_norm = X_norm;
    mdl.actuals.y_norm = y_norm;
    mdl.predictions = pred;

for resp = 1:size(y_norm,2)

    SS_res = sum((pred(:,resp) - y_norm(:,resp)).^2);
    SS_tot = sum((y_norm(:,resp) - mean(y_norm(:,resp))).^2);

    mdl.performance.R2_per_response(resp) = ...
        1 - SS_res/SS_tot;

    % Pearson's r
    mdl.performance.R_per_response(resp) = ...
        corr(pred(:,resp), y_norm(:,resp), ...
        'Type', 'Pearson', ...
        'Rows', 'complete');

    % Spearman's rho
    mdl.performance.Spearman_per_response(resp) = ...
        corr(pred(:,resp), y_norm(:,resp), ...
        'Type', 'Spearman', ...
        'Rows', 'complete');
end

mdl.performance.R2_overall = ...
    mean(mdl.performance.R2_per_response, 'omitnan');

mdl.performance.R_overall = ...
    mean(mdl.performance.R_per_response, 'omitnan');

mdl.performance.Spearman_overall = ...
    mean(mdl.performance.Spearman_per_response, 'omitnan');
end

function regressed_data = regress_site_effects(data, site_data)
    % REGRESS_SITE_EFFECTS  Remove recruitment-site mean effects via dummy coding.
    %
    %   regressed_data = regress_site_effects(data, site_data)
    %
    %   Builds (n_sites - 1) dummy indicator columns for site membership
    %   (the last site is the implicit reference/omitted category), fits an
    %   OLS regression of DATA on these dummies via the backslash operator,
    %   and returns the residuals. This removes additive site-mean
    %   differences from DATA (columns of predictors or outcomes) prior to
    %   PLS fitting, so that PLS components are not driven by between-site
    %   scanner/recruitment differences.
    %
    %   INPUTS
    %     data       - [n_subjects x n_vars] matrix to be site-corrected
    %     site_data  - [n_subjects x 1] integer site labels (1..n_sites)
    %
    %   OUTPUT
    %     regressed_data - residuals of DATA after removing site mean effects
    n_sites = length(unique(site_data));
    site_dummies = zeros(length(site_data), n_sites-1);
    for i = 1:n_sites-1
        site_dummies(:,i) = (site_data == i);
    end
    regressed_data = data - site_dummies * (site_dummies \ data);
end

function mdl = pls_multivariate_nestedCV(X, y, confounds, n_repeats)
% PLS_MULTIVARIATE_NESTEDCV  Nested cross-validated PLS regression with
% confound (site) regression, used to estimate out-of-sample predictive
% performance and select the number of PLS components.
%
%   mdl = pls_multivariate_nestedCV(X, y, confounds, n_repeats)
%
% PURPOSE
%   For each of N_REPEATS repeats, performs a 5-fold outer cross-validation
%   split. Within each outer training fold, a 3-fold x 10-repeat inner CV
%   loop is used to select the number of PLS components (1 up to
%   min(n_predictors, n_responses, 4)) that maximizes mean out-of-fold R².
%   A final PLS model is then trained on the full outer-training fold
%   (using the selected number of components) and evaluated on the held-out
%   outer-test fold. All folds/repeats use confound (site) regression via
%   multivariate regression (mvregress, with a least-squares fallback) and
%   z-scoring, fit on the training data and applied to test data to avoid
%   leakage.
%
% INPUTS
%   X          - [n_subjects x n_predictors] predictor matrix
%   y          - [n_subjects x n_responses] outcome matrix
%   confounds  - [n_subjects x n_confounds] confound/site matrix (may be a
%                cell array, in which case it is converted via cell2mat).
%                In this pipeline this is the recruitment-site label.
%   n_repeats  - number of independent repeats of the 5-fold outer CV
%                (each repeat re-draws the fold assignment)
%
% OUTPUT
%   mdl        - 1 x n_repeats struct array. For repeat i:
%     mdl(i).betas                     - [n_preds x n_responses x k_outer] PLS
%                                         beta coefficients (intercept excluded)
%                                         for each of the 5 outer folds
%     mdl(i).optimal_components        - [k_outer x 1] number of components
%                                         selected by inner CV, per outer fold
%     mdl(i).loadings.X / .Y           - per-fold X/Y loadings (cell arrays)
%     mdl(i).scores.X / .Y             - per-fold X/Y scores (cell arrays)
%     mdl(i).performance.train.overall_R / overall_R2
%                                       - training-fold fit quality, pooled
%                                         across all 5 outer folds
%     mdl(i).performance.test.overall_R / overall_R2
%                                       - mean held-out R/R² across the 5
%                                         outer folds (primary CV performance metric)
%     mdl(i).performance.test.fold_test_R2 / fold_test_R
%                                       - per-fold held-out R²/R ([k_outer x 1])
%     mdl(i).data.train.train_predictions / train_actuals
%                                       - subject-level training predictions/actuals
%                                         (site/confound-residualized, z-scored units)
%     mdl(i).data.test.test_predictions / test_actuals
%                                       - subject-level held-out predictions/actuals
%     mdl(i).data.fold_train_idx{fold} / fold_test_idx{fold}
%                                       - subject indices used in each outer fold
%
% NOTES
%   - Reproducibility: rng(0,'twister') is set once at the start, so the
%     sequence of fold assignments (crossvalind) is deterministic given
%     n_repeats and the data size.
%   - The first block of code (the "demonstrating fold 1" section) is a
%     leftover illustrative example that prints train/test set sizes for a
%     single fold; it does not feed into the returned mdl output and can be
%     ignored/removed without affecting results.
%   - Confound regression is performed twice per outer fold: once inside
%     the inner CV loop (for choosing the component count) and again when
%     training the final model on the full outer-training fold.

rng(0, 'twister'); % always initiate for reproducibility

n_subjects = length(y);
% Set up cross-validation
k_folds = 5;  % Number of folds
cv_indices = crossvalind('Kfold', n_subjects, k_folds);

% --- Illustrative-only block ---------------------------------------
% Demonstrates a single train/test split for fold 1 and prints its sizes.
% This does not affect the nested-CV results computed below; it is purely
% diagnostic output.
fold = 1;  % Pick any fold to demonstrate

fprintf('Demonstrating fold %d of %d:\n', fold, k_folds);

% Split data into train and test sets
test_idx = (cv_indices == fold);    % Logical index for test set
train_idx = ~test_idx;              % Logical index for train set

% Extract the actual data splits
X_train = X(train_idx, :);
X_test = X(test_idx, :);
y_train = y(train_idx, :);
y_test = y(test_idx, :);

% Show what we got
fprintf('Training set size: %d subjects\n', sum(train_idx));
fprintf('Test set size: %d subjects\n', sum(test_idx));
% --- End illustrative-only block ------------------------------------

n_subjects = size(X, 1);
n_responses = size(y, 2);
k_outer = 5;   % outer CV folds (performance estimation)
k_inner = 3;   % inner CV folds (component selection)

% Check if confounds are in cell format
if iscell(confounds)
    confounds_numeric = cell2mat(confounds);
else
    confounds_numeric = confounds;
end
confounds_numeric = double(confounds_numeric);

n_preds = size(X, 2);
% Cap the number of PLS components considered at 4 (or fewer if limited
% by the number of predictors/responses).
max_components = min([n_preds, n_responses, 4]);

% Initialize storage
all_predictions = zeros(n_subjects, n_responses, n_repeats);
true_values = zeros(n_subjects, n_responses, n_repeats);
betas = zeros(n_preds, n_responses, k_outer, n_repeats);
optimal_comps_all = zeros(k_outer, n_repeats);

% Pre-allocate mdl structure
mdl(n_repeats) = struct();
for i = 1:n_repeats
    mdl(i).betas = [];
    mdl(i).optimal_components = [];
    mdl(i).performance = struct('train', struct(), 'test', struct());
    mdl(i).data = struct('train', struct(), 'test', struct());
    mdl(i).loadings = struct('X', [], 'Y', []);
    mdl(i).scores = struct('X', [], 'Y', []);
end

% Cross-validation
fprintf('Starting PLS CV with %d repeats, %d outer folds\n', n_repeats, k_outer);
for repeat = 1:n_repeats
    fprintf('Repeat %d/%d: ', repeat, n_repeats);
    % Initialize for this repeat
    subject_train_predictions = zeros(n_subjects, n_responses);
    subject_train_actuals = zeros(n_subjects, n_responses);
    train_predictions = [];
    train_actuals = [];
    fold_test_R2 = [];
    fold_test_R = [];
    fold_test_Spearman = [];
    subject_predictions = zeros(n_subjects, n_responses);
    subject_actuals = zeros(n_subjects, n_responses);
    X_loadings_folds = cell(k_outer, 1);
    Y_loadings_folds = cell(k_outer, 1);
    X_scores_folds = cell(k_outer, 1);
    Y_scores_folds = cell(k_outer, 1);
    
    % OUTER CV LOOP: re-draw a fresh 5-fold split for this repeat, used to
    % estimate out-of-sample (held-out) performance.
    cv_indices = crossvalind('Kfold', n_subjects, k_outer);
    
    for fold = 1:k_outer
        fprintf('F%d ', fold);
        % Get outer train/test split
        outer_test_idx = (cv_indices == fold);
        outer_train_idx = ~outer_test_idx;
        curr_test_idx = find(outer_test_idx);
        train_idx_nums = find(outer_train_idx);
        
        X_outer_train = X(outer_train_idx, :);
        X_outer_test = X(outer_test_idx, :);
        y_outer_train = y(outer_train_idx, :);
        y_outer_test = y(outer_test_idx, :);
        D_outer_train = confounds_numeric(outer_train_idx, :);
        D_outer_test = confounds_numeric(outer_test_idx, :);

        

        
        % INNER CV LOOP: Hyperparameter tuning
        % For each candidate number of PLS components (1..max_components),
        % repeatedly (n_inner_repeats times) run 3-fold CV within the outer
        % training fold and average the held-out R² across all inner
        % folds/repeats. The component count with the highest average R²
        % is selected for this outer fold.
        n_inner_repeats = 10;
        component_performance = zeros(max_components, 1);
        
        for comp = 1:max_components
            all_inner_R2s = [];
            
            for inner_repeat = 1:n_inner_repeats
                inner_cv_indices = crossvalind('Kfold', sum(outer_train_idx), k_inner);
                
                for inner_fold = 1:k_inner
                    % Inner train/test split
                    inner_test_idx = (inner_cv_indices == inner_fold);
                    inner_train_idx = ~inner_test_idx;
                    
                    X_inner_train = X_outer_train(inner_train_idx, :);
                    X_inner_test = X_outer_train(inner_test_idx, :);
                    y_inner_train = y_outer_train(inner_train_idx, :);
                    y_inner_test = y_outer_train(inner_test_idx, :);
                    D_inner_train = D_outer_train(inner_train_idx, :);
                    D_inner_test = D_outer_train(inner_test_idx, :);
                    
                    % NORMALIZE confounds, X, Y
                    % Mean/SD are computed on the inner-training data only
                    % and then applied to both inner-train and inner-test
                    % data, preventing test-set information leakage.
                    D_train_mean = mean(D_inner_train, 1);
                    D_train_std = std(D_inner_train, 1);
                    D_train_std(D_train_std < eps) = eps;  % avoid divide-by-zero
                    D_inner_train_norm = (D_inner_train - D_train_mean) ./ D_train_std;
                    D_inner_test_norm = (D_inner_test - D_train_mean) ./ D_train_std;
                    
                    X_train_mean = mean(X_inner_train, 1);
                    X_train_std = std(X_inner_train, 1);
                    X_train_std(X_train_std < eps) = eps;
                    X_inner_train_norm = (X_inner_train - X_train_mean) ./ X_train_std;
                    X_inner_test_norm = (X_inner_test - X_train_mean) ./ X_train_std;
                    
                    y_train_mean = mean(y_inner_train, 1);
                    y_train_std = std(y_inner_train, 1);
                    y_train_std(y_train_std < eps) = eps;
                    y_inner_train_norm = (y_inner_train - y_train_mean) ./ y_train_std;
                    y_inner_test_norm = (y_inner_test - y_train_mean) ./ y_train_std;
                    
                    % CONFOUND REGRESSION on normalized data using multivariate regression
                    % Regress X (and separately y) on the confound(s), fit
                    % on inner-training data, and subtract the confound-
                    % predicted component from both inner-train and
                    % inner-test data to obtain confound-residualized data.
                    try
                        [B_X, ~] = mvregress(D_inner_train_norm, X_inner_train_norm);
                        X_inner_train_resid = X_inner_train_norm - D_inner_train_norm * B_X;
                        X_inner_test_resid = X_inner_test_norm - D_inner_test_norm * B_X;
                    catch
                        % Fallback to matrix division if mvregress fails
                        B_X = D_inner_train_norm \ X_inner_train_norm;
                        X_inner_train_resid = X_inner_train_norm - D_inner_train_norm * B_X;
                        X_inner_test_resid = X_inner_test_norm - D_inner_test_norm * B_X;
                    end

                    try
                        [B_Y, ~] = mvregress(D_inner_train_norm, y_inner_train_norm);
                        y_inner_train_resid = y_inner_train_norm - D_inner_train_norm * B_Y;
                        y_inner_test_resid = y_inner_test_norm - D_inner_test_norm * B_Y;
                    catch
                        % Fallback to matrix division if mvregress fails
                        B_Y = D_inner_train_norm \ y_inner_train_norm;
                        y_inner_train_resid = y_inner_train_norm - D_inner_train_norm * B_Y;
                        y_inner_test_resid = y_inner_test_norm - D_inner_test_norm * B_Y;
                    end

                    % PLS MODEL: fit on inner-train residuals with COMP
                    % components, predict inner-test residuals, and compute
                    % held-out R² pooled across all outcomes/subjects in
                    % this inner fold.
                    [~,~,~,~,beta_inner] = plsregress(X_inner_train_resid, y_inner_train_resid, comp);
                    inner_pred = [ones(size(X_inner_test_resid,1),1), X_inner_test_resid] * beta_inner;
                    
                    % R² for whole fold
                    SS_res = sum((inner_pred(:) - y_inner_test_resid(:)).^2);
                    SS_tot = sum((y_inner_test_resid(:) - mean(y_inner_test_resid(:))).^2);
                    
                    if SS_tot > eps
                        inner_R2 = 1 - SS_res/SS_tot;
                    else
                        inner_R2 = NaN;
                    end
                    
                    all_inner_R2s = [all_inner_R2s; inner_R2];
                end
            end
            
            component_performance(comp) = mean(all_inner_R2s, 'omitnan');
        end
        
        % Select optimal number of components = the one with highest mean
        % inner-CV R² for this outer fold.
        [~, optimal_components] = max(component_performance);
        optimal_comps_all(fold, repeat) = optimal_components;
        
        % FINAL MODEL TRAINING ON OUTER DATA
        % Repeat the same normalization + confound-regression procedure,
        % this time fit on the full outer-training fold and applied to the
        % held-out outer-test fold, using the OPTIMAL_COMPONENTS chosen above.
        % NORMALIZE confounds, X, Y
        D_train_mean = mean(D_outer_train, 1);
        D_train_std = std(D_outer_train, 1);
        D_train_std(D_train_std < eps) = eps;
        D_outer_train_norm = (D_outer_train - D_train_mean) ./ D_train_std;
        D_outer_test_norm = (D_outer_test - D_train_mean) ./ D_train_std;
        
        X_train_mean = mean(X_outer_train, 1);
        X_train_std = std(X_outer_train, 1);
        X_train_std(X_train_std < eps) = eps;
        X_outer_train_norm = (X_outer_train - X_train_mean) ./ X_train_std;
        X_outer_test_norm = (X_outer_test - X_train_mean) ./ X_train_std;
        
        y_train_mean = mean(y_outer_train, 1);
        y_train_std = std(y_outer_train, 1);
        y_train_std(y_train_std < eps) = eps;
        y_outer_train_norm = (y_outer_train - y_train_mean) ./ y_train_std;
        y_outer_test_norm = (y_outer_test - y_train_mean) ./ y_train_std;
        
        % CONFOUND REGRESSION on normalized data using multivariate regression
        try
            [B_X, ~] = mvregress(D_outer_train_norm, X_outer_train_norm);
            X_outer_train_resid = X_outer_train_norm - D_outer_train_norm * B_X;
            X_outer_test_resid = X_outer_test_norm - D_outer_test_norm * B_X;
        catch
            % Fallback to matrix division if mvregress fails
            B_X = D_outer_train_norm \ X_outer_train_norm;
            X_outer_train_resid = X_outer_train_norm - D_outer_train_norm * B_X;
            X_outer_test_resid = X_outer_test_norm - D_outer_test_norm * B_X;
        end

        try
            [B_Y, ~] = mvregress(D_outer_train_norm, y_outer_train_norm);
            y_outer_train_resid = y_outer_train_norm - D_outer_train_norm * B_Y;
            y_outer_test_resid = y_outer_test_norm - D_outer_test_norm * B_Y;
        catch
            % Fallback to matrix division if mvregress fails
            B_Y = D_outer_train_norm \ y_outer_train_norm;
            y_outer_train_resid = y_outer_train_norm - D_outer_train_norm * B_Y;
            y_outer_test_resid = y_outer_test_norm - D_outer_test_norm * B_Y;
        end

        % TRAIN FINAL PLS MODEL for this outer fold
        [XL, YL, XS, YS, beta] = plsregress(X_outer_train_resid, y_outer_train_resid, optimal_components);
        
        % Store results
        X_loadings_folds{fold} = XL;
        Y_loadings_folds{fold} = YL;
        X_scores_folds{fold} = XS;
        Y_scores_folds{fold} = YS;
        full_beta = beta(2:end, :);  % drop the intercept row before storing
        betas(:,:,fold,repeat) = full_beta;
        
        % Predictions on the outer-training fold (in-sample, for reference)
        train_pred = [ones(size(X_outer_train_resid,1),1), X_outer_train_resid] * beta;
        train_predictions = [train_predictions; train_pred];
        train_actuals = [train_actuals; y_outer_train_resid];
        
        % Predictions on the held-out outer-test fold (the actual
        % cross-validated performance estimate)
        test_pred = [ones(size(X_outer_test_resid,1),1), X_outer_test_resid] * beta;
        
        % Performance metrics - R² and R for whole fold
        SS_res = sum((test_pred(:) - y_outer_test_resid(:)).^2);
        SS_tot = sum((y_outer_test_resid(:) - mean(y_outer_test_resid(:))).^2);
        
        if SS_tot > eps
            fold_R2 = 1 - SS_res/SS_tot;
        else
            fold_R2 = NaN;
        end
        
        % Pearson's R for whole fold
fold_R = corr( ...
    test_pred(:), ...
    y_outer_test_resid(:), ...
    'Type', 'Pearson', ...
    'Rows', 'complete');

% Spearman's rho for whole fold
fold_Spearman = corr( ...
    test_pred(:), ...
    y_outer_test_resid(:), ...
    'Type', 'Spearman', ...
    'Rows', 'complete');

fold_test_R2 = [fold_test_R2; fold_R2];
fold_test_R = [fold_test_R; fold_R];
fold_test_Spearman = [fold_test_Spearman; fold_Spearman];        
        % Store predictions
        subject_train_predictions(train_idx_nums, :) = train_pred;
        subject_train_actuals(train_idx_nums, :) = y_outer_train_resid;
        all_predictions(curr_test_idx, :, repeat) = test_pred;
        true_values(curr_test_idx, :, repeat) = y_outer_test_resid;
        subject_predictions(curr_test_idx, :) = test_pred;
        subject_actuals(curr_test_idx, :) = y_outer_test_resid;

        % Store fold indices (subject indices, not logical masks)
        mdl(repeat).data.fold_train_idx{fold} = train_idx_nums;
        mdl(repeat).data.fold_test_idx{fold} = curr_test_idx;
    end
    
    % Simple averages across the 5 outer folds for this repeat
    test_R2_overall = mean(fold_test_R2, 'omitnan');
    test_R_overall = mean(fold_test_R, 'omitnan');
    test_Spearman_overall = mean(fold_test_Spearman, 'omitnan');
    
    % Training metrics, pooled across all outer-training folds
    train_R_overall = corr(train_predictions(:), train_actuals(:));
    train_R2_overall = 1 - sum((train_predictions(:) - train_actuals(:)).^2) / ...
                            sum((train_actuals(:) - mean(train_actuals(:))).^2);
    
    % Store results for this repeat
    mdl(repeat).betas = betas(:,:,:,repeat);
    mdl(repeat).optimal_components = optimal_comps_all(:,repeat);
    mdl(repeat).loadings.X = X_loadings_folds;
    mdl(repeat).loadings.Y = Y_loadings_folds;
    mdl(repeat).scores.X = X_scores_folds;
    mdl(repeat).scores.Y = Y_scores_folds;
    
    mdl(repeat).performance.train.overall_R = train_R_overall;
    mdl(repeat).performance.train.overall_R2 = train_R2_overall;
    
    mdl(repeat).performance.test.overall_R = test_R_overall;
mdl(repeat).performance.test.overall_R2 = test_R2_overall;
mdl(repeat).performance.test.overall_Spearman = test_Spearman_overall;

mdl(repeat).performance.test.fold_test_R2 = fold_test_R2;
mdl(repeat).performance.test.fold_test_R = fold_test_R;
mdl(repeat).performance.test.fold_test_Spearman = fold_test_Spearman;
    
    mdl(repeat).data.train.train_predictions = subject_train_predictions;
    mdl(repeat).data.train.train_actuals = subject_train_actuals;
    mdl(repeat).data.test.test_predictions = subject_predictions;
    mdl(repeat).data.test.test_actuals = subject_actuals;
    train_idx_nums = find(outer_train_idx);

    fprintf('\n');
end

fprintf('PLS CV complete\n');

end