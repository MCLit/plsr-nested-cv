% ESTIMATE_PLS_MODEL_ADDITIONAL_BENEFIT_OF_COG_FULLNESTED
%
% PURPOSE
%   Fully-nested version of the cognition-added-value permutation test.
%   Unlike estimate_pls_model_additional_benefit_of_cog.m, which fits
%   each permutation at a FIXED number of components (final_ncomps from
%   the real data), this version re-selects the number of components
%   (1-4) via a fresh inner cross-validation loop within every outer
%   fold of every permutation - i.e. it exactly mirrors the component-
%   selection logic in pls_multivariate_nestedCV (estimate_pls_model.m),
%   just repeated once per permutation instead of once on the real data.
%
%   This is the "fully nested" extension noted as an option in the
%   fixed-ncomps version's header comment.
%
% RUNTIME WARNING (read before running)
%   The fixed-ncomps version costs n_perms x 2 models x k_folds PLS fits.
%   This version costs roughly that, TIMES (k_inner x n_inner_repeats x
%   max_components) per outer fold, because every outer fold now runs
%   its own inner-CV component search. At the values used in the main
%   pipeline (k_inner=3, n_inner_repeats=10, max_components=4), that is
%   roughly a 120x multiplier per fold. Measured on this dataset: ~7.1 s
%   per permutation serially (n_perms=1000 -> ~118 min serial). Runs via
%   PARFOR - actual wall time depends on your worker count; the script
%   prints an updated estimate at both serial and parallel speed before
%   the full run starts, and reports actual elapsed time at the end.
%
%   REPRODUCIBILITY UNDER PARFOR: parfor iterations do not execute in a
%   fixed order, so a single rng() call before the loop would NOT give
%   reproducible per-permutation results (which permutation gets which
%   random draw would depend on worker scheduling). Instead, each
%   permutation is given its own independent random substream keyed to
%   the permutation INDEX (RandStream('mrg32k3a'), Substream = perm),
%   so results are identical regardless of worker count or completion
%   order. If no parallel pool is available, parfor degrades to serial
%   execution automatically - no code change needed either way.
%
% PIPELINE POSITION
%   Run AFTER estimate_pls_model.m (requires FINAL_MODELS.MAT).
%   Produces COGNITION_PERMUTATION_RESULTS_FULLNESTED.MAT - a separate
%   file from the fixed-ncomps version's output, so neither overwrites
%   the other. results.m does not currently load this file; add a
%   loading block analogous to the existing Part 8 if you want it
%   reported there instead of (or alongside) the fixed-ncomps result.
%
% INPUT FILE (must be in the current working directory)
%   final_models.mat
%       Produced by estimate_pls_model.m. Only models 4 (Threat/Dep) and
%       5 (Threat/Dep+Cog) are used.
%
% OUTPUT FILE
%   cognition_permutation_results_fullnested.mat
%       Same fields as the fixed-ncomps version's output, plus
%       ncomps_selected_base / ncomps_selected_full, which record the
%       component count chosen at each outer fold of each permutation
%       (useful for sanity-checking that selection is stable and not
%       just bouncing around noisily).
%
% DEPENDENCIES
%   MATLAB Statistics and Machine Learning Toolbox

clear;
clc;

load('final_models.mat');   % provides all_models_combined, Y_labels, model_names

% Note: no single rng() call here - see REPRODUCIBILITY UNDER PARFOR below.
% Each permutation sets its own reproducible substream individually.

n_perms = 5000;              % measured ~7.1 s/permutation serially on this data (~118 min serial total)
k_folds = 5;
k_inner = 3;
n_inner_repeats = 10;        % lower than the main pipeline's 10 - see runtime warning
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

max_components_base = min([size(X_base, 2), size(y, 2), max_components]);
max_components_full = min([size(X_full, 2), size(y, 2), max_components]);

%% --- Parallel pool setup -------------------------------------------------
%  On CSF3 (Slurm), the number of cores available is given by the
%  SLURM_NTASKS environment variable set by the jobscript's #SBATCH -n
%  line. We size the pool to that explicitly, per CSF3's documented
%  practice - relying on parpool's default sizing can over- or
%  under-subscribe the node relative to what Slurm actually allocated.
%  When SLURM_NTASKS is not set (e.g. running interactively, not via
%  sbatch), this falls back to gcp/parpool's default local pool.

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

%% --- Observed dR2, computed directly from final_models.mat -------------
%  Uses the existing nested-CV results (final_ncomps chosen on the real
%  data), same as the fixed-ncomps version - this part does not change,
%  since it is not part of the permutation loop.

n_repeats_cv = min(numel(mdl_base.cv_results), numel(mdl_full.cv_results));
base_R2 = nan(1, n_repeats_cv);
full_R2 = nan(1, n_repeats_cv);

for r = 1:n_repeats_cv
    base_R2(r) = mdl_base.cv_results(r).performance.test.overall_R2;
    full_R2(r) = mdl_full.cv_results(r).performance.test.overall_R2;
end

valid = ~isnan(base_R2) & ~isnan(full_R2);
observed_diffs = full_R2(valid) - base_R2(valid);

observed_dR2_mean = mean(observed_diffs);
observed_dR2_sd = std(observed_diffs);
observed_dR2_ci = prctile(observed_diffs, [2.5, 97.5]);

fprintf('Observed dR2 (Threat/Dep+Cog - Threat/Dep): %.5f +/- %.5f\n', ...
    observed_dR2_mean, observed_dR2_sd);
fprintf('Observed dR2 95%% percentile interval: [%.5f, %.5f]\n', ...
    observed_dR2_ci(1), observed_dR2_ci(2));
fprintf('(from %d paired nested-CV repeats)\n', numel(observed_diffs));

%% --- Timing pilot (serial, to estimate total runtime before committing) --

n_pilot = min(3, n_perms);
fprintf('\nTiming %d pilot permutation(s) (serial) to estimate total runtime...\n', n_pilot);

pilot_times = nan(n_pilot, 1);
for p = 1:n_pilot
    t_pilot = tic;

    pilot_stream = RandStream('mrg32k3a', 'Seed', 0);
    pilot_stream.Substream = p;
    RandStream.setGlobalStream(pilot_stream);

    perm_order = randperm(n_subjects);
    X_full_perm_pilot = X_full;
    X_full_perm_pilot(:, cognitive_cols) = X_full(perm_order, cognitive_cols);
    fold_assignment_pilot = crossvalind('Kfold', n_subjects, k_folds);

    eval_nested_cv(X_base, y, site_data, k_folds, fold_assignment_pilot, ...
        max_components_base, k_inner, n_inner_repeats);
    eval_nested_cv(X_full_perm_pilot, y, site_data, k_folds, fold_assignment_pilot, ...
        max_components_full, k_inner, n_inner_repeats);

    pilot_times(p) = toc(t_pilot);
end

RandStream.setGlobalStream(RandStream('mrg32k3a', 'Seed', 0));  % reset for main run

mean_pilot_time = mean(pilot_times);
est_serial_total = mean_pilot_time * n_perms;
est_parallel_total = est_serial_total / n_workers;

fprintf('Mean time per permutation (serial): %.1f s\n', mean_pilot_time);
fprintf('Estimated total runtime: %.1f min serial, ~%.1f min with %d workers\n', ...
    est_serial_total / 60, est_parallel_total / 60, n_workers);

%% --- Build permutation null (fully nested: components re-selected) -----
%  Each permutation gets its own independent random substream, keyed to
%  the permutation index (not to execution order), so results are
%  reproducible regardless of worker count or the order iterations
%  finish in - this matters specifically because parfor iterations do
%  NOT execute in order, so a single rng() seed set once before the loop
%  would not give reproducible per-permutation results the way it does
%  in a plain for loop.

fprintf('\nRunning %d permutations (fully nested component selection)...\n', n_perms);

null_dR2 = nan(n_perms, 1);
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

    [R2_base_perm, ncomps_base_this] = eval_nested_cv(X_base, y, site_data, ...
        k_folds, fold_assignment, max_components_base, k_inner, n_inner_repeats);

    [R2_full_perm, ncomps_full_this] = eval_nested_cv(X_full_perm, y, site_data, ...
        k_folds, fold_assignment, max_components_full, k_inner, n_inner_repeats);

    null_dR2(perm) = R2_full_perm - R2_base_perm;
    ncomps_selected_base(perm, :) = ncomps_base_this;
    ncomps_selected_full(perm, :) = ncomps_full_this;

    if mod(perm, 50) == 0
        fprintf('  permutation %d / %d done\n', perm, n_perms);
    end
end

fprintf('Permutations complete. Actual elapsed time: %.1f min\n', toc(t_main) / 60);

%% --- Compare observed to null --------------------------------------------

null_dR2 = null_dR2(~isnan(null_dR2));

null_mean = mean(null_dR2);
null_sd = std(null_dR2);

p_one_tailed = (sum(null_dR2 >= observed_dR2_mean) + 1) / (numel(null_dR2) + 1);

p_two_tailed = (sum(abs(null_dR2 - null_mean) >= abs(observed_dR2_mean - null_mean)) + 1) / ...
    (numel(null_dR2) + 1);

percentile = mean(null_dR2 < observed_dR2_mean) * 100;

fprintf('\n--- Results (fully nested) ---\n');
fprintf('Null dR2:      %.5f +/- %.5f  (n_perms = %d)\n', null_mean, null_sd, numel(null_dR2));
fprintf('Observed dR2:  %.5f  (percentile %.1f%% of null distribution)\n', ...
    observed_dR2_mean, percentile);
fprintf('p (one-tailed, H1: cognition adds value): %.4f\n', p_one_tailed);
fprintf('p (two-tailed, centered on null mean):     %.4f\n', p_two_tailed);

fprintf('\nComponent selection stability (mode / range across all perms x folds):\n');
fprintf('  Base model (Threat/Dep):      mode = %d, range = [%d, %d]\n', ...
    mode(ncomps_selected_base(:)), min(ncomps_selected_base(:)), max(ncomps_selected_base(:)));
fprintf('  Full model (Threat/Dep+Cog):  mode = %d, range = [%d, %d]\n', ...
    mode(ncomps_selected_full(:)), min(ncomps_selected_full(:)), max(ncomps_selected_full(:)));

%% --- Save results ----------------------------------------------------------

save('cognition_permutation_results_fullnested_5k.mat', ...
    'observed_dR2_mean', 'observed_dR2_sd', 'observed_dR2_ci', 'observed_diffs', ...
    'null_dR2', 'null_mean', 'null_sd', ...
    'percentile', 'p_one_tailed', 'p_two_tailed', ...
    'ncomps_selected_base', 'ncomps_selected_full', ...
    'n_perms', 'k_folds', 'k_inner', 'n_inner_repeats', 'max_components');

fprintf('\nSaved to cognition_permutation_results_fullnested.mat\n');


%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function [R2, ncomps_per_fold] = eval_nested_cv(X, y, site_data, k_folds, ...
        fold_assignment, max_components, k_inner, n_inner_repeats)
    % EVAL_NESTED_CV  One pass of outer k-fold CV, with the number of PLS
    % components re-selected via inner CV within each outer fold - i.e.
    % the same nested logic as pls_multivariate_nestedCV, but run once
    % here rather than accumulated across repeats.
    %
    %   [R2, ncomps_per_fold] = eval_nested_cv(X, y, site_data, k_folds, ...
    %       fold_assignment, max_components, k_inner, n_inner_repeats)
    %
    %   OUTPUTS
    %     R2              - pooled (across all outer folds) test-set R2
    %     ncomps_per_fold - [k_folds x 1] component count selected in
    %                       each outer fold, for stability checking

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
end