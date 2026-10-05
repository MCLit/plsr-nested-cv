% ESTIMATE_PLS_MODEL_ADDED_BENEFIT_COGNITION_INDIVIDUAL_OUTCOMES
%
% PURPOSE
%   Individual-outcome version of the cognition added-value permutation
%   test (Threat/Dep+Cog vs Threat/Dep). For each mental health outcome
%   separately, tests whether adding the cognitive predictor block
%   improves out-of-sample R2 beyond what is expected by chance.
%
%   The permutation procedure is identical to
%   estimate_pls_model_added_benefit_cognition_extended_par.m (same
%   permuted cognitive block, same fully nested component selection, same
%   per-permutation random substreams), except that R2 is also tracked
%   PER OUTCOME instead of only pooled across outcomes.
%
% OUTPUT FILES (saved in this order, so uncorrected results exist on disk
% before any post-processing happens)
%   1. cognition_permutation_individual_outcomes_uncorrected.mat
%        Observed and null dR2 per outcome, uncorrected one-tailed
%        p-values. Saved immediately after the permutation loop.
%   2. cognition_permutation_individual_outcomes_fdr.mat
%        Everything above plus Benjamini-Hochberg FDR-adjusted p-values
%        (q-values) across the outcomes.
%
% OBSERVED dR2 PER OUTCOME
%   Computed from the stored cv_results predictions (no model refitting).
%   For each of the 100 nested-CV repeats, R2 is computed per outcome on
%   the subject-level predictions pooled across the 5 outer folds, for
%   each model; dR2 = R2(Threat/Dep+Cog) - R2(Threat/Dep). The observed
%   statistic is the mean over repeats. This matches how the null R2 is
%   computed (pooled across folds), so the two sides are comparable.
%
% REPRODUCIBILITY CHECK
%   The random substream logic is unchanged, so the POOLED null dR2 from
%   this run should reproduce the pooled null from the earlier run
%   (cognition_permutation_results_fullnested.mat) exactly, provided
%   n_inner_repeats, k_inner, max_components and the data are unchanged.
%   If that file is in the working directory, the script compares them
%   and reports the result.
%
% RUNTIME
%   Same as the pooled version (per-outcome R2 is bookkeeping on
%   predictions that are already generated, not extra model fitting).
%
% DEPENDENCIES
%   MATLAB Statistics and Machine Learning Toolbox, Parallel Computing
%   Toolbox (falls back to serial if unavailable)

clear;
clc;

load('final_models.mat');   % provides all_models_combined, Y_labels, model_names

n_perms = 5000;
k_folds = 5;
k_inner = 3;
n_inner_repeats = 10;        % must match the earlier pooled run for the reproducibility check
max_components = 4;

%% --- Pull model data --------------------------------------------------

mdl_base = all_models_combined{4};   % Threat/Dep
mdl_full = all_models_combined{5};   % Threat/Dep+Cog

X_base = mdl_base.whole_data.X;      % [age, sex, threat, deprivation]
X_full = mdl_full.whole_data.X;      % [age, sex, threat, deprivation, 11 cognitive cols]

y = mdl_full.whole_data.y;
site_data = mdl_full.whole_data.site_data;

assert(isequal(size(mdl_base.whole_data.y), size(mdl_full.whole_data.y)), ...
    'Base and full models do not share the same sample - permutation pairing invalid.');
assert(isequal(mdl_base.whole_data.site_data, mdl_full.whole_data.site_data), ...
    'Base and full models do not share the same site assignments - permutation pairing invalid.');

cognitive_cols = 5:15;  % columns of X_full corresponding to the 11 CANTAB measures
assert(size(X_full, 2) == 15, ...
    'X_full does not have 15 columns as expected - check cognitive_cols indexing.');

n_subjects = size(X_full, 1);
n_outcomes = size(y, 2);

assert(numel(Y_labels) == n_outcomes, ...
    'Y_labels (%d) does not match the number of outcome columns in y (%d).', ...
    numel(Y_labels), n_outcomes);

max_components_base = min([size(X_base, 2), n_outcomes, max_components]);
max_components_full = min([size(X_full, 2), n_outcomes, max_components]);

%% --- Parallel pool setup -------------------------------------------------
%  Size the pool from SLURM_NTASKS when running under Slurm (CSF3
%  practice); otherwise fall back to the default local pool.

slurm_ntasks = getenv('SLURM_NTASKS');

try
    pool = gcp('nocreate');
    if isempty(pool)
        if ~isempty(slurm_ntasks)
            n_requested = str2double(slurm_ntasks);
            pool = parpool(n_requested);
            fprintf('Running under Slurm - started pool with %d workers (SLURM_NTASKS).\n', n_requested);
        else
            pool = parpool;
            fprintf('Not running under Slurm - started pool with default local worker count.\n');
        end
    end
    n_workers = pool.NumWorkers;
    fprintf('Pool ready: %d workers.\n', n_workers);
catch
    n_workers = 1;
    warning('Could not start a parallel pool - running serially.');
end

%% --- Observed dR2 per outcome, from stored cv_results -------------------

n_repeats_cv = min(numel(mdl_base.cv_results), numel(mdl_full.cv_results));

observed_R2_base = nan(n_repeats_cv, n_outcomes);
observed_R2_full = nan(n_repeats_cv, n_outcomes);

for r = 1:n_repeats_cv
    observed_R2_base(r, :) = per_outcome_R2( ...
        mdl_base.cv_results(r).data.test.test_predictions, ...
        mdl_base.cv_results(r).data.test.test_actuals);

    observed_R2_full(r, :) = per_outcome_R2( ...
        mdl_full.cv_results(r).data.test.test_predictions, ...
        mdl_full.cv_results(r).data.test.test_actuals);
end

observed_dR2_outcomes = observed_R2_full - observed_R2_base;     % [n_repeats x n_outcomes]
observed_dR2_mean = mean(observed_dR2_outcomes, 1, 'omitnan');   % [1 x n_outcomes]
observed_dR2_sd = std(observed_dR2_outcomes, 0, 1, 'omitnan');
observed_dR2_ci = prctile(observed_dR2_outcomes, [2.5, 97.5], 1); % [2 x n_outcomes]

fprintf('\nObserved dR2 per outcome (Threat/Dep+Cog - Threat/Dep), mean over %d repeats:\n', n_repeats_cv);
for o = 1:n_outcomes
    fprintf('  %-25s %9.5f +/- %.5f   95%% interval [%9.5f, %9.5f]\n', ...
        Y_labels{o}, observed_dR2_mean(o), observed_dR2_sd(o), ...
        observed_dR2_ci(1, o), observed_dR2_ci(2, o));
end

%% --- Timing pilot (serial, to estimate total runtime before committing) --

n_pilot = min(3, n_perms);
fprintf('\nTiming %d pilot permutation(s) (serial) to estimate total runtime...\n', n_pilot);

pilot_times = nan(n_pilot, 1);
for pp = 1:n_pilot
    t_pilot = tic;

    pilot_stream = RandStream('mrg32k3a', 'Seed', 0);
    pilot_stream.Substream = pp;
    RandStream.setGlobalStream(pilot_stream);

    perm_order = randperm(n_subjects);
    X_full_perm_pilot = X_full;
    X_full_perm_pilot(:, cognitive_cols) = X_full(perm_order, cognitive_cols);
    fold_assignment_pilot = crossvalind('Kfold', n_subjects, k_folds);

    eval_nested_cv(X_base, y, site_data, k_folds, fold_assignment_pilot, ...
        max_components_base, k_inner, n_inner_repeats);
    eval_nested_cv(X_full_perm_pilot, y, site_data, k_folds, fold_assignment_pilot, ...
        max_components_full, k_inner, n_inner_repeats);

    pilot_times(pp) = toc(t_pilot);
end

RandStream.setGlobalStream(RandStream('mrg32k3a', 'Seed', 0));  % reset for main run

mean_pilot_time = mean(pilot_times);
est_serial_total = mean_pilot_time * n_perms;
est_parallel_total = est_serial_total / n_workers;

fprintf('Mean time per permutation (serial): %.1f s\n', mean_pilot_time);
fprintf('Estimated total runtime: %.1f min serial, ~%.1f min with %d workers\n', ...
    est_serial_total / 60, est_parallel_total / 60, n_workers);

%% --- Build permutation null (fully nested: components re-selected) -----
%  Each permutation gets its own independent random substream keyed to the
%  permutation index, so results are reproducible regardless of worker
%  count or completion order. The order of random draws inside each
%  iteration (randperm, crossvalind, then base and full model evaluation)
%  is deliberately identical to the pooled version of this script.

fprintf('\nRunning %d permutations (fully nested component selection)...\n', n_perms);

null_dR2 = nan(n_perms, 1);                      % pooled across outcomes (for the reproducibility check)
null_dR2_outcomes = nan(n_perms, n_outcomes);    % per outcome
ncomps_selected_base = nan(n_perms, k_folds);
ncomps_selected_full = nan(n_perms, k_folds);

t_main = tic;

parfor perm = 1:n_perms

    perm_stream = RandStream('mrg32k3a', 'Seed', 0);
    perm_stream.Substream = perm;
    RandStream.setGlobalStream(perm_stream);

    perm_order = randperm(n_subjects);
    X_full_perm = X_full;
    X_full_perm(:, cognitive_cols) = X_full(perm_order, cognitive_cols);

    fold_assignment = crossvalind('Kfold', n_subjects, k_folds);

    [R2_base_perm, ncomps_base_this, R2o_base] = eval_nested_cv(X_base, y, site_data, ...
        k_folds, fold_assignment, max_components_base, k_inner, n_inner_repeats);

    [R2_full_perm, ncomps_full_this, R2o_full] = eval_nested_cv(X_full_perm, y, site_data, ...
        k_folds, fold_assignment, max_components_full, k_inner, n_inner_repeats);

    null_dR2(perm) = R2_full_perm - R2_base_perm;
    null_dR2_outcomes(perm, :) = R2o_full - R2o_base;
    ncomps_selected_base(perm, :) = ncomps_base_this;
    ncomps_selected_full(perm, :) = ncomps_full_this;

    if mod(perm, 100) == 0
        fprintf('  permutation %d / %d done\n', perm, n_perms);
    end
end

fprintf('Permutations complete. Actual elapsed time: %.1f min\n', toc(t_main) / 60);

%% --- Reproducibility check against the earlier pooled run ---------------

prev_file = 'cognition_permutation_results_fullnested.mat';

if isfile(prev_file)
    prev = load(prev_file, 'null_dR2');
    n_compare = min(numel(prev.null_dR2), numel(null_dR2));
    max_abs_diff = max(abs(prev.null_dR2(1:n_compare) - null_dR2(1:n_compare)), [], 'omitnan');

    fprintf('\nReproducibility check vs %s (first %d permutations):\n', prev_file, n_compare);
    fprintf('  max |difference| in pooled null dR2 = %.3g\n', max_abs_diff);

    if max_abs_diff < 1e-10
        fprintf('  OK - pooled null reproduced exactly.\n');
    else
        fprintf(['  WARNING - pooled null does not match the earlier run. Check that\n' ...
            '  n_inner_repeats, k_inner, max_components and the input data are\n' ...
            '  identical to the earlier run (or that the MATLAB version is the same).\n']);
    end
else
    fprintf('\nReproducibility check skipped (%s not found in working directory).\n', prev_file);
end

%% --- Uncorrected results (SAVED FIRST) ---------------------------------

null_mean_outcomes = mean(null_dR2_outcomes, 1, 'omitnan');
null_sd_outcomes = std(null_dR2_outcomes, 0, 1, 'omitnan');

p_uncorrected = nan(1, n_outcomes);
percentile_outcomes = nan(1, n_outcomes);

for o = 1:n_outcomes
    null_o = null_dR2_outcomes(:, o);
    null_o = null_o(~isnan(null_o));

    p_uncorrected(o) = (sum(null_o >= observed_dR2_mean(o)) + 1) / (numel(null_o) + 1);
    percentile_outcomes(o) = mean(null_o < observed_dR2_mean(o)) * 100;
end

save_vars = {'Y_labels', ...
    'observed_dR2_outcomes', 'observed_dR2_mean', 'observed_dR2_sd', 'observed_dR2_ci', ...
    'null_dR2_outcomes', 'null_dR2', 'null_mean_outcomes', 'null_sd_outcomes', ...
    'p_uncorrected', 'percentile_outcomes', ...
    'ncomps_selected_base', 'ncomps_selected_full', ...
    'n_perms', 'k_folds', 'k_inner', 'n_inner_repeats', 'max_components'};

save('cognition_permutation_individual_outcomes_uncorrected.mat', save_vars{:});

fprintf('\nUncorrected results saved to cognition_permutation_individual_outcomes_uncorrected.mat\n');

fprintf('\n--- Individual outcomes: uncorrected (one-tailed, H1: cognition adds value) ---\n');
fprintf('%-25s %10s %10s %10s %12s %10s\n', ...
    'Outcome', 'Obs dR2', 'Null mean', 'Null SD', 'Percentile', 'p (unc.)');
for o = 1:n_outcomes
    fprintf('%-25s %10.5f %10.5f %10.5f %11.1f%% %10.4f\n', ...
        Y_labels{o}, observed_dR2_mean(o), null_mean_outcomes(o), ...
        null_sd_outcomes(o), percentile_outcomes(o), p_uncorrected(o));
end

%% --- FDR correction (Benjamini-Hochberg across outcomes) ---------------

q_fdr = bh_fdr(p_uncorrected);
significant_fdr = q_fdr < 0.05;

save_vars_fdr = [save_vars, {'q_fdr', 'significant_fdr'}];

save('cognition_permutation_individual_outcomes_fdr.mat', save_vars_fdr{:});

fprintf('\n--- Individual outcomes: BH-FDR across %d outcomes ---\n', n_outcomes);
fprintf('%-25s %10s %10s %10s %14s\n', ...
    'Outcome', 'Obs dR2', 'p (unc.)', 'q (FDR)', 'FDR q < 0.05');
for o = 1:n_outcomes
    if significant_fdr(o)
        flag = 'YES';
    else
        flag = 'no';
    end
    fprintf('%-25s %10.5f %10.4f %10.4f %14s\n', ...
        Y_labels{o}, observed_dR2_mean(o), p_uncorrected(o), q_fdr(o), flag);
end

fprintf('\nFDR-corrected results saved to cognition_permutation_individual_outcomes_fdr.mat\n');


%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function R2_per_outcome = per_outcome_R2(pred, actual)
    % PER_OUTCOME_R2  R2 computed separately for each column (outcome).
    %
    %   pred, actual - [n x n_outcomes]
    %   R2_per_outcome - [1 x n_outcomes]; NaN where an outcome has no
    %                    variance in the actuals

    n_out = size(actual, 2);
    R2_per_outcome = nan(1, n_out);

    for o = 1:n_out
        SS_res = sum((pred(:, o) - actual(:, o)).^2);
        SS_tot = sum((actual(:, o) - mean(actual(:, o))).^2);

        if SS_tot > eps
            R2_per_outcome(o) = 1 - SS_res / SS_tot;
        end
    end
end

function q = bh_fdr(p)
    % BH_FDR  Benjamini-Hochberg adjusted p-values (q-values).
    %
    %   q = bh_fdr(p) returns adjusted p-values in the same order as the
    %   input, controlling the false discovery rate across all tests in p.

    p = p(:);
    m = numel(p);

    [p_sorted, order] = sort(p);
    q_sorted = p_sorted .* m ./ (1:m)';
    q_sorted = flipud(cummin(flipud(q_sorted)));
    q_sorted = min(q_sorted, 1);

    q = nan(m, 1);
    q(order) = q_sorted;
    q = q';
end

function [R2, ncomps_per_fold, R2_per_outcome] = eval_nested_cv(X, y, site_data, k_folds, ...
        fold_assignment, max_components, k_inner, n_inner_repeats)
    % EVAL_NESTED_CV  One pass of outer k-fold CV, with the number of PLS
    % components re-selected via inner CV within each outer fold - i.e.
    % the same nested logic as pls_multivariate_nestedCV, but run once
    % here rather than accumulated across repeats.
    %
    %   [R2, ncomps_per_fold, R2_per_outcome] = eval_nested_cv(X, y, ...
    %       site_data, k_folds, fold_assignment, max_components, ...
    %       k_inner, n_inner_repeats)
    %
    %   OUTPUTS
    %     R2              - pooled (across all outer folds and outcomes)
    %                       test-set R2
    %     ncomps_per_fold - [k_folds x 1] component count selected in
    %                       each outer fold, for stability checking
    %     R2_per_outcome  - [1 x n_outcomes] test-set R2 per outcome,
    %                       pooled across the outer folds

    all_test_pred = [];
    all_test_actual = [];
    ncomps_per_fold = nan(k_folds, 1);

    for fold = 1:k_folds

        outer_test_idx = (fold_assignment == fold);
        outer_train_idx = ~outer_test_idx;

        X_outer_train = X(outer_train_idx, :);
        X_outer_test = X(outer_test_idx, :);
        y_outer_train = y(outer_train_idx, :);
        y_outer_test = y(outer_test_idx, :);
        D_outer_train = site_data(outer_train_idx, :);
        D_outer_test = site_data(outer_test_idx, :);

        % --- INNER CV: select number of components (fresh normalization
        % and confound regression within each inner split, matching
        % pls_multivariate_nestedCV's inner loop exactly) ---
        component_performance = zeros(max_components, 1);

        for comp = 1:max_components

            all_inner_R2s = [];

            for inner_repeat = 1:n_inner_repeats

                inner_cv_indices = crossvalind('Kfold', sum(outer_train_idx), k_inner);

                for inner_fold = 1:k_inner

                    inner_test_idx = (inner_cv_indices == inner_fold);
                    inner_train_idx = ~inner_test_idx;

                    X_inner_train = X_outer_train(inner_train_idx, :);
                    X_inner_test = X_outer_train(inner_test_idx, :);
                    y_inner_train = y_outer_train(inner_train_idx, :);
                    y_inner_test = y_outer_train(inner_test_idx, :);
                    D_inner_train = D_outer_train(inner_train_idx, :);
                    D_inner_test = D_outer_train(inner_test_idx, :);

                    D_mean = mean(D_inner_train, 1); D_std = std(D_inner_train, 1);
                    D_std(D_std < eps) = eps;
                    D_inner_train_n = (D_inner_train - D_mean) ./ D_std;
                    D_inner_test_n = (D_inner_test - D_mean) ./ D_std;

                    X_mean = mean(X_inner_train, 1); X_std = std(X_inner_train, 1);
                    X_std(X_std < eps) = eps;
                    X_inner_train_n = (X_inner_train - X_mean) ./ X_std;
                    X_inner_test_n = (X_inner_test - X_mean) ./ X_std;

                    y_mean = mean(y_inner_train, 1); y_std = std(y_inner_train, 1);
                    y_std(y_std < eps) = eps;
                    y_inner_train_n = (y_inner_train - y_mean) ./ y_std;
                    y_inner_test_n = (y_inner_test - y_mean) ./ y_std;

                    try
                        [B_X, ~] = mvregress(D_inner_train_n, X_inner_train_n);
                        X_inner_train_resid = X_inner_train_n - D_inner_train_n * B_X;
                        X_inner_test_resid = X_inner_test_n - D_inner_test_n * B_X;
                    catch
                        B_X = D_inner_train_n \ X_inner_train_n;
                        X_inner_train_resid = X_inner_train_n - D_inner_train_n * B_X;
                        X_inner_test_resid = X_inner_test_n - D_inner_test_n * B_X;
                    end

                    try
                        [B_Y, ~] = mvregress(D_inner_train_n, y_inner_train_n);
                        y_inner_train_resid = y_inner_train_n - D_inner_train_n * B_Y;
                        y_inner_test_resid = y_inner_test_n - D_inner_test_n * B_Y;
                    catch
                        B_Y = D_inner_train_n \ y_inner_train_n;
                        y_inner_train_resid = y_inner_train_n - D_inner_train_n * B_Y;
                        y_inner_test_resid = y_inner_test_n - D_inner_test_n * B_Y;
                    end

                    [~, ~, ~, ~, beta_inner] = plsregress(X_inner_train_resid, y_inner_train_resid, comp);
                    inner_pred = [ones(size(X_inner_test_resid, 1), 1), X_inner_test_resid] * beta_inner;

                    SS_res = sum((inner_pred(:) - y_inner_test_resid(:)).^2);
                    SS_tot = sum((y_inner_test_resid(:) - mean(y_inner_test_resid(:))).^2);

                    if SS_tot > eps
                        inner_R2 = 1 - SS_res / SS_tot;
                    else
                        inner_R2 = NaN;
                    end

                    all_inner_R2s = [all_inner_R2s; inner_R2]; %#ok<AGROW>
                end
            end

            component_performance(comp) = mean(all_inner_R2s, 'omitnan');
        end

        [~, optimal_components] = max(component_performance);
        ncomps_per_fold(fold) = optimal_components;

        % --- Fit final model for this outer fold at the selected
        % component count, using the outer train/test split ---
        D_mean = mean(D_outer_train, 1); D_std = std(D_outer_train, 1);
        D_std(D_std < eps) = eps;
        D_outer_train_n = (D_outer_train - D_mean) ./ D_std;
        D_outer_test_n = (D_outer_test - D_mean) ./ D_std;

        X_mean = mean(X_outer_train, 1); X_std = std(X_outer_train, 1);
        X_std(X_std < eps) = eps;
        X_outer_train_n = (X_outer_train - X_mean) ./ X_std;
        X_outer_test_n = (X_outer_test - X_mean) ./ X_std;

        y_mean = mean(y_outer_train, 1); y_std = std(y_outer_train, 1);
        y_std(y_std < eps) = eps;
        y_outer_train_n = (y_outer_train - y_mean) ./ y_std;
        y_outer_test_n = (y_outer_test - y_mean) ./ y_std;

        try
            [B_X, ~] = mvregress(D_outer_train_n, X_outer_train_n);
            X_outer_train_resid = X_outer_train_n - D_outer_train_n * B_X;
            X_outer_test_resid = X_outer_test_n - D_outer_test_n * B_X;
        catch
            B_X = D_outer_train_n \ X_outer_train_n;
            X_outer_train_resid = X_outer_train_n - D_outer_train_n * B_X;
            X_outer_test_resid = X_outer_test_n - D_outer_test_n * B_X;
        end

        try
            [B_Y, ~] = mvregress(D_outer_train_n, y_outer_train_n);
            y_outer_train_resid = y_outer_train_n - D_outer_train_n * B_Y;
            y_outer_test_resid = y_outer_test_n - D_outer_test_n * B_Y;
        catch
            B_Y = D_outer_train_n \ y_outer_train_n;
            y_outer_train_resid = y_outer_train_n - D_outer_train_n * B_Y;
            y_outer_test_resid = y_outer_test_n - D_outer_test_n * B_Y;
        end

        [~, ~, ~, ~, beta] = plsregress(X_outer_train_resid, y_outer_train_resid, optimal_components);
        pred = [ones(size(X_outer_test_resid, 1), 1), X_outer_test_resid] * beta;

        all_test_pred = [all_test_pred; pred]; %#ok<AGROW>
        all_test_actual = [all_test_actual; y_outer_test_resid]; %#ok<AGROW>
    end

    SS_res = sum((all_test_pred(:) - all_test_actual(:)).^2);
    SS_tot = sum((all_test_actual(:) - mean(all_test_actual(:))).^2);

    if SS_tot > eps
        R2 = 1 - SS_res / SS_tot;
    else
        R2 = NaN;
    end

    R2_per_outcome = per_outcome_R2(all_test_pred, all_test_actual);
end
