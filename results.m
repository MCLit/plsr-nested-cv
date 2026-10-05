%% MASTER RESULTS SCRIPT (CONSOLE-ONLY)
%
%  Loads fitted models and null models. Part numbers below match the
%  %% PART headers in the body exactly.
%   
%    Part 0 - Setup
%    Part 1 - Whole-sample predictive performance
%    Part 2 - Betas (Supplementary Tables 3-5)
%    Part 3 - Individual-outcome whole-sample performance (Supplementary
%               table 6)
%    Part 4 - Model generalisability (cross-validated) & Supplementary
%               table 2
%    Part 5 - Overall model performance vs chance & Supplementary
%               table 7
%    Part 6 - Individual-outcome generalisability descriptives (Supplementary table 8)
%    Part 7 - Individual-outcomes vs chance (Supplementary table 9)
%    Part 8 - Model comparisons added value of cognition
%    Part 9 - Model components, loadings and VIP scores
%             (variance explained, X-loadings + VIP, Y-loadings)
%    Part 10 - Model comparisons added value of cognition for individual
%               outcomes
%
%  Model comparisons between Threat/Dep, Cognition and Threat/Dep+Cog are
%  reported via the permutation test in Part 8 only. An earlier version
%  of this script also ran independent-samples t-tests (ttest2) on
%  repeated-CV R2 values for these comparisons; that approach understates
%  variance because CV repeats are not independent (Dietterich, 1998;
%  Nadeau & Bengio, 2003), so it has been removed rather than reported
%  alongside the correct result.
%
%  Pearson's R and Spearman's rho are reported alongside R2 throughout.
%
%  ALL RESULTS ARE PRINTED TO THE CONSOLE ONLY - no Excel files are
%  written by this script.
%
%  REQUIRED FILES (must be in the current working directory)
%    final_models.mat                    - from estimate_pls_model.m
%    null_mdl/null_models_5models.mat    - null model fits
%    cognition_permutation_results.mat   - from
%      estimate_pls_model_additional_benefit_of_cog.m (Part 8 only;
%      script will warn and skip Part 8 if this is missing, all other
%      parts run regardless)
%  ========================================================================

%% PART 0 - SETUP

clear;
clc;

load('final_models.mat');
load(fullfile(pwd, 'null_mdl', 'null_models_5models.mat'));

% Model information
model_names = { ...
    'Trauma', ...
    'Cognition', ...
    'Trauma+Cog', ...
    'Threat/Dep', ...
    'Threat/Dep+Cog'};

% Predictor names
predictor_names = cell(5,1);

predictor_names{1} = { ...
    'age', 'sex', ...
    'emotional_abuse', 'physical_abuse', 'sexual_abuse', ...
    'emotional_neglect', 'physical_neglect'};

predictor_names{2} = { ...
    'age', 'sex', ...
    'AGN_latency_pos', 'AGN_latency_neg', ...
    'CGT_delay_aversion', 'CGT_deliberation', 'CGT_bet', ...
    'CGT_quality', 'CGT_risk_adj', 'CGT_risk_taking', ...
    'RPV_attention', 'SWM_errors', 'SWM_strategy'};

predictor_names{3} = { ...
    'age', 'sex', ...
    'emotional_abuse', 'physical_abuse', 'sexual_abuse', ...
    'emotional_neglect', 'physical_neglect', ...
    'AGN_latency_pos', 'AGN_latency_neg', ...
    'CGT_delay_aversion', 'CGT_deliberation', 'CGT_bet', ...
    'CGT_quality', 'CGT_risk_adj', 'CGT_risk_taking', ...
    'RPV_attention', 'SWM_errors', 'SWM_strategy'};

predictor_names{4} = { ...
    'age', 'sex', 'threat', 'deprivation'};

predictor_names{5} = { ...
    'age', 'sex', 'threat', 'deprivation', ...
    'AGN_latency_pos', 'AGN_latency_neg', ...
    'CGT_delay_aversion', 'CGT_deliberation', ...
    'CGT_bet', 'CGT_quality', 'CGT_risk_adj', ...
    'CGT_risk_taking', 'RPV_attention', ...
    'SWM_errors', 'SWM_strategy'};

% Outcome labels
outcome_labels = { ...
    'Specific Phobia', ...
    'Social Phobia', ...
    'Agoraphobia', ...
    'OCD', ...
    'Anxiety', ...
    'Depression', ...
    'PTSD', ...
    'Conduct Problems', ...
    'Hyperactivity/Inattention'};

n_models   = numel(model_names);
n_outcomes = numel(outcome_labels);
n_repeats  = 100;



%% PART 1: WHOLE-SAMPLE PREDICTIVE PERFORMANCE

fprintf('\n\n============================================================\n');
fprintf('WHOLE-SAMPLE MODEL PERFORMANCE\n');
fprintf('============================================================\n');

model_R2 = nan(n_models, n_repeats);
model_R  = nan(n_models, n_repeats);
model_Spearman = nan(n_models, n_repeats);

for model_idx = 1:n_models

    if isempty(all_models_combined{model_idx})
        continue;
    end

    mdl = all_models_combined{model_idx};

    R2 = mdl.performance.R2_overall;
    R  = mdl.performance.R_overall;
    Spearman = mdl.performance.Spearman_overall;

    n_comps = mdl.final_ncomps;

    fprintf(['%-20s R2 = %8.4f   Pearson R = %8.4f   ' ...
        'Spearman rho = %8.4f   Components = %d\n'], ...
        model_names{model_idx}, ...
        R2, ...
        R, ...
        Spearman, ...
        n_comps);

    % Cross-validation performance
    n_available_repeats = min( ...
        n_repeats, ...
        numel(mdl.cv_results));

    for repeat = 1:n_available_repeats

        model_R2(model_idx, repeat) = ...
            mdl.cv_results(repeat).performance.test.overall_R2;

        model_R(model_idx, repeat) = ...
            mdl.cv_results(repeat).performance.test.overall_R;

        if isfield(mdl.cv_results(repeat).performance.test, ...
                'overall_Spearman')

            model_Spearman(model_idx, repeat) = ...
                mdl.cv_results(repeat).performance.test.overall_Spearman;
        end
    end
end

%% PART 2: BETAS (Supplementary Tables 3 - 5)


fprintf('\n\n============================================================\n');
fprintf('BETA WEIGHTS\n');
fprintf('============================================================\n');

for i = 1:5

    beta = all_models_combined{i}.beta;

    % Ensure predictors are rows
    if size(beta,1) ~= length(predictor_names{i})
        beta = beta';
    end

    fprintf('\n\n============================================================\n');
    fprintf('MODEL: %s\n', model_names{i});
    fprintf('============================================================\n');

    for j = 1:size(beta,1)

        % Print predictor name if available
        if j <= length(predictor_names{i})
            fprintf('%-30s', predictor_names{i}{j});
        else
            fprintf('%-30s', sprintf('Row %d', j));
        end

        % Print every beta value that actually exists
        for k = 1:size(beta,2)
            fprintf('  %.6f', beta(j,k));
        end

        fprintf('\n');
    end

end

%% PART 3: INDIVIDUAL OUTCOME WHOLE-SAMPLE PERFORMANCE (Supplementary Table 6)

fprintf('\n\n============================================================\n');
fprintf('INDIVIDUAL OUTCOME WHOLE-SAMPLE PERFORMANCE\n');
fprintf('============================================================\n');

for outcome_idx = 1:n_outcomes

    fprintf('\n%-30s', outcome_labels{outcome_idx});

    for model_idx = 1:n_models

        if isempty(all_models_combined{model_idx})
            continue;
        end

        mdl = all_models_combined{model_idx};

        R2 = mdl.performance.R2_per_response(outcome_idx);

        R = mdl.performance.R_per_response(outcome_idx);

        Spearman = ...
            mdl.performance.Spearman_per_response(outcome_idx);

        fprintf([' | %s: R2 = %.3f, Pearson R = %.3f, ' ...
            'Spearman rho = %.3f'], ...
            model_names{model_idx}, ...
            R2, ...
            R, ...
            Spearman);
    end
end


%% PART 4: MODEL GENERALISABILITY & Supplementary Table 2


fprintf('\n\n============================================================\n');
fprintf('MODEL GENERALISABILITY\n');
fprintf('============================================================\n');

for model_idx = 1:n_models

    valid_R2 = model_R2( ...
        model_idx, ...
        ~isnan(model_R2(model_idx,:)));

    valid_R = model_R( ...
        model_idx, ...
        ~isnan(model_R(model_idx,:)));

    valid_Spearman = model_Spearman( ...
        model_idx, ...
        ~isnan(model_Spearman(model_idx,:)));

    if isempty(valid_R2)
        continue;
    end

    mean_R2 = mean(valid_R2);
    sd_R2   = std(valid_R2);

    mean_R = mean(valid_R);
    sd_R   = std(valid_R);

    if isempty(valid_Spearman)
        mean_Spearman = NaN;
        sd_Spearman = NaN;
    else
        mean_Spearman = mean(valid_Spearman);
        sd_Spearman = std(valid_Spearman);
    end

    fprintf(['%-20s R2 = %.3f +/- %.3f   ' ...
        'Pearson R = %.3f +/- %.3f   ' ...
        'Spearman rho = %.3f +/- %.3f\n'], ...
        model_names{model_idx}, ...
        mean_R2, sd_R2, ...
        mean_R, sd_R, ...
        mean_Spearman, sd_Spearman);
end



%% PART 5: OVERALL MODEL PERFORMANCE VS CHANCE & Supplementary Table 7

fprintf('\n\n============================================================\n');
fprintf('OVERALL MODEL PERFORMANCE VS CHANCE\n');
fprintf('============================================================\n');

null_R2_overall = nan(n_models, n_repeats);

for model_idx = 1:n_models

    if isempty(all_null_models{model_idx})
        continue;
    end

    n_null_repeats = min( ...
        n_repeats, ...
        numel(all_null_models{model_idx}.cv_results));

    for repeat = 1:n_null_repeats

        null_R2_overall(model_idx, repeat) = ...
            all_null_models{model_idx}.cv_results(repeat) ...
            .performance.test.overall_R2;
    end
end

for model_idx = 1:n_models

    empirical = model_R2(model_idx, :);
    empirical = empirical(~isnan(empirical));

    null_dist = null_R2_overall(model_idx, :);
    null_dist = null_dist(~isnan(null_dist));

    if isempty(empirical) || isempty(null_dist)
        continue;
    end

    empirical_mean = mean(empirical);
    empirical_sd   = std(empirical);

    null_mean = mean(null_dist);
    null_sd   = std(null_dist);

    p_value = ...
        (sum(null_dist >= empirical_mean) + 1) / ...
        (numel(null_dist) + 1);

    percentile = (1 - p_value) * 100;

    above_chance = p_value < 0.05;

    if above_chance
        result = 'YES';
    else
        result = 'NO';
    end

    fprintf('\n%s\n', model_names{model_idx});
    fprintf('  Empirical R2: %.3f +/- %.3f\n', ...
        empirical_mean, empirical_sd);
    fprintf('  Null R2:      %.3f +/- %.3f\n', ...
        null_mean, null_sd);
    fprintf('  p = %.4f\n', p_value);
    fprintf('  Percentile = %.1f%%\n', percentile);
    fprintf('  Above chance = %s\n', result);
end

%% PART 6: INDIVIDUAL OUTCOME GENERALISABILITY DESCRIPTIVES (Supplementary table 8)

fprintf('\n\n============================================================\n');
fprintf('INDIVIDUAL OUTCOME GENERALISABILITY\n');
fprintf('============================================================\n');

outcome_R2_all_repeats = nan( ...
    n_models, n_outcomes, n_repeats);

outcome_R_all_repeats = nan( ...
    n_models, n_outcomes, n_repeats);

outcome_Spearman_all_repeats = nan( ...
    n_models, n_outcomes, n_repeats);

for model_idx = 1:n_models

    if isempty(all_models_combined{model_idx})
        continue;
    end

    mdl = all_models_combined{model_idx};

    n_available_repeats = min( ...
        n_repeats, numel(mdl.cv_results));

    for repeat = 1:n_available_repeats

        test_preds = ...
            mdl.cv_results(repeat).data.test.test_predictions;

        test_actuals = ...
            mdl.cv_results(repeat).data.test.test_actuals;

        for outcome_idx = 1:n_outcomes

            predictions = test_preds(:, outcome_idx);
            actuals     = test_actuals(:, outcome_idx);

            valid = ~isnan(predictions) & ~isnan(actuals);

            predictions = predictions(valid);
            actuals     = actuals(valid);

            if isempty(predictions)
                continue;
            end

            % R2

            SS_res = sum((predictions - actuals).^2);

            SS_tot = sum((actuals - mean(actuals)).^2);

            if SS_tot > eps

                outcome_R2_all_repeats( ...
                    model_idx, outcome_idx, repeat) = ...
                    1 - SS_res / SS_tot;

            else

                outcome_R2_all_repeats( ...
                    model_idx, outcome_idx, repeat) = 0;
            end

            % Pearson's R

            if numel(predictions) >= 2 && ...
                    std(predictions) > 0 && ...
                    std(actuals) > 0

                outcome_R_all_repeats( ...
                    model_idx, outcome_idx, repeat) = ...
                    corr( ...
                    predictions, ...
                    actuals, ...
                    'Type', 'Pearson', ...
                    'Rows', 'complete');

                % Spearman's rho

                outcome_Spearman_all_repeats( ...
                    model_idx, outcome_idx, repeat) = ...
                    corr( ...
                    predictions, ...
                    actuals, ...
                    'Type', 'Spearman', ...
                    'Rows', 'complete');
            end
        end
    end
end

for outcome_idx = 1:n_outcomes

    fprintf('\n%-30s', outcome_labels{outcome_idx});

    for model_idx = 1:n_models

        performance_R2 = squeeze( ...
            outcome_R2_all_repeats( ...
            model_idx, outcome_idx, :));

        performance_R = squeeze( ...
            outcome_R_all_repeats( ...
            model_idx, outcome_idx, :));

        performance_Spearman = squeeze( ...
            outcome_Spearman_all_repeats( ...
            model_idx, outcome_idx, :));

        performance_R2 = ...
            performance_R2(~isnan(performance_R2));

        performance_R = ...
            performance_R(~isnan(performance_R));

        performance_Spearman = ...
            performance_Spearman(~isnan(performance_Spearman));

        if isempty(performance_R2)
            continue;
        end

        mean_R2 = mean(performance_R2);
        sd_R2   = std(performance_R2);

        if isempty(performance_R)
            mean_R = NaN;
            sd_R = NaN;
        else
            mean_R = mean(performance_R);
            sd_R = std(performance_R);
        end

        if isempty(performance_Spearman)
            mean_Spearman = NaN;
            sd_Spearman = NaN;
        else
            mean_Spearman = mean(performance_Spearman);
            sd_Spearman = std(performance_Spearman);
        end

        fprintf([' | %s: R2 = %.3f +/- %.3f, ' ...
            'Pearson R = %.3f +/- %.3f, ' ...
            'Spearman rho = %.3f +/- %.3f'], ...
            model_names{model_idx}, ...
            mean_R2, sd_R2, ...
            mean_R, sd_R, ...
            mean_Spearman, sd_Spearman);
    end
end



%% PART 7: INDIVIDUAL OUTCOMES VS CHANCE (Supplementary table 9)

fprintf('\n\n============================================================\n');
fprintf('INDIVIDUAL OUTCOMES VS CHANCE\n');
fprintf('============================================================\n');

null_outcome_R2_all_repeats = nan( ...
    n_models, n_outcomes, n_repeats);

for model_idx = 1:n_models

    if isempty(all_null_models{model_idx})
        continue;
    end

    n_available_repeats = min( ...
        n_repeats, ...
        numel(all_null_models{model_idx}.cv_results));

    for repeat = 1:n_available_repeats

        null_R2 = ...
            all_null_models{model_idx} ...
            .cv_results(repeat) ...
            .performance.test.R2_per_response;

        n_available_outcomes = min( ...
            n_outcomes, numel(null_R2));

        null_outcome_R2_all_repeats( ...
            model_idx, 1:n_available_outcomes, repeat) = ...
            null_R2(1:n_available_outcomes);
    end
end

for outcome_idx = 1:n_outcomes

    fprintf('\n%s\n', outcome_labels{outcome_idx});

    for model_idx = 1:n_models

        empirical = squeeze( ...
            outcome_R2_all_repeats( ...
            model_idx, outcome_idx, :));

        empirical = empirical(~isnan(empirical));

        null_dist = squeeze( ...
            null_outcome_R2_all_repeats( ...
            model_idx, outcome_idx, :));

        null_dist = null_dist(~isnan(null_dist));

        if isempty(empirical) || isempty(null_dist)
            continue;
        end

        empirical_mean = mean(empirical);
        empirical_sd   = std(empirical);

        null_mean = mean(null_dist);
        null_sd   = std(null_dist);

        p_value = ...
            (sum(null_dist >= empirical_mean) + 1) / ...
            (numel(null_dist) + 1);

        percentile = (1 - p_value) * 100;

        above_chance = p_value < 0.05;

        if above_chance
            result = 'YES';
        else
            result = 'NO';
        end

        fprintf('%-20s | Empirical = %.3f +/- %.3f | ', ...
            model_names{model_idx}, ...
            empirical_mean, ...
            empirical_sd);

        fprintf('Null = %.3f +/- %.3f | p = %.4f | %s\n', ...
            null_mean, ...
            null_sd, ...
            p_value, ...
            result);
    end
end



%% PART 8: MODEL COMPARISONS ADDED VALUE OF COGNITION (precomputed
%  permutation test) Supplementary tables 10 and 11

%  Loads cognition_permutation_results.mat, produced once by
%  estimate_pls_model_additional_benefit_of_cog.m. Console output only -
%  this script does not re-run the permutation test.

fprintf('\n\n============================================================\n');
fprintf('ADDED VALUE OF COGNITION (Threat/Dep+Cog vs Threat/Dep)\n');
fprintf('Permutation test - precomputed results\n');
fprintf('============================================================\n');

if ~isfile('cognition_permutation_results.mat')

    warning(['cognition_permutation_results.mat not found. Run ' ...
        'estimate_pls_model_additional_benefit_of_cog.m once to ' ...
        'generate it, then re-run this script.']);

else

    load('cognition_permutation_results.mat');

    fprintf('Observed dR2:  %.5f +/- %.5f  (n = %d paired repeats)\n', ...
        observed_dR2_mean, observed_dR2_sd, numel(observed_diffs));
    fprintf('Observed dR2 95%% percentile interval: [%.5f, %.5f]\n', ...
        observed_dR2_ci(1), observed_dR2_ci(2));

    fprintf('\nPermutation null: %.5f +/- %.5f  (n_perms = %d)\n', ...
        null_mean, null_sd, n_perms);
    fprintf('Observed dR2 percentile within null distribution: %.1f%%\n', ...
        percentile);

    fprintf('\np (one-tailed, H1: cognition adds value): %.4f  <- primary result\n', ...
        p_one_tailed);
    fprintf('p (two-tailed, centered on null mean):     %.4f\n', p_two_tailed);

    if p_one_tailed < 0.05
        fprintf(['\nResult: cognition''s added predictive value exceeds ' ...
            'chance (permutation p < 0.05).\n']);
    else
        fprintf(['\nResult: cognition''s added predictive value does NOT ' ...
            'exceed chance (permutation p >= 0.05).\n']);
    end

end

%% PART 9: MODEL STRUCTURE - MODEL COMPONENTS, LOADINGS AND VIP

fprintf('\n\n============================================================\n');
fprintf('MODEL COMPONENTS, LOADINGS AND VIP SCORES\n');
fprintf('============================================================\n');

for model_idx = 1:n_models

    if isempty(all_models_combined{model_idx})
        continue;
    end

    mdl = all_models_combined{model_idx};
    n_comps = mdl.final_ncomps;

    fprintf('\n============================================================\n');
    fprintf('MODEL: %s\n', model_names{model_idx});
    fprintf('============================================================\n');

    % Variance explained

    fprintf('\nVariance Explained by Components:\n');
    fprintf('%-15s %15s %15s\n', ...
        'Component', 'X Variance', 'Y Variance');
    fprintf('%s\n', repmat('-', 1, 48));

    for comp = 1:n_comps

        x_var = mdl.PCTVAR(1, comp);
        y_var = mdl.PCTVAR(2, comp);

        fprintf('%-15s %14.2f%% %14.2f%%\n', ...
            sprintf('Component %d', comp), ...
            x_var, y_var);
    end

    cumulative_x = sum(mdl.PCTVAR(1, 1:n_comps));
    cumulative_y = sum(mdl.PCTVAR(2, 1:n_comps));

    fprintf('%-15s %14.2f%% %14.2f%%\n', ...
        'Cumulative', ...
        cumulative_x, ...
        cumulative_y);

    % X-loadings and VIP

    fprintf('\nX-Loadings and VIP Scores:\n');
    fprintf('%-25s', 'Predictor');

    for comp = 1:n_comps
        fprintf('%12s', sprintf('Comp%d', comp));
    end

    fprintf('%12s\n', 'VIP');

    for pred_idx = 1:numel(predictor_names{model_idx})

        predictor = predictor_names{model_idx}{pred_idx};

        fprintf('%-25s', predictor);

        for comp = 1:n_comps

            loading = mdl.x_loadings(pred_idx, comp);

            fprintf('%12.4f', loading);
        end

        vip = mdl.vip_scores(pred_idx);

        fprintf('%12.4f\n', vip);
    end

    % Y-loadings

    fprintf('\nY-Loadings:\n');
    fprintf('%-25s', 'Outcome');

    for comp = 1:n_comps
        fprintf('%12s', sprintf('Comp%d', comp));
    end

    fprintf('\n');

    for outcome_idx = 1:n_outcomes

        outcome = outcome_labels{outcome_idx};

        fprintf('%-25s', outcome);

        for comp = 1:n_comps

            loading = mdl.y_loadings(outcome_idx, comp);

            fprintf('%12.4f', loading);
        end

        fprintf('\n');
    end

end

%% PART 10: INDIVIDUAL OUTCOMES - ADDED VALUE OF COGNITION (precomputed
%  permutation test)

%  Loads cognition_permutation_individual_outcomes_fdr.mat, produced once by
%  estimate_pls_model_added_benefit_cognition_individual_outcomes.m. Console
%  output only - this script does not re-run the permutation test.

fprintf('\n\n============================================================\n');
fprintf('INDIVIDUAL OUTCOMES: ADDED VALUE OF COGNITION (Threat/Dep+Cog vs Threat/Dep)\n');
fprintf('Permutation test - precomputed results\n');
fprintf('============================================================\n');

outcome_perm_file = 'cognition_permutation_individual_outcomes_fdr.mat';

if ~isfile(outcome_perm_file)

    warning(['%s not found. Run ' ...
        'estimate_pls_model_added_benefit_cognition_individual_outcomes.m ' ...
        'once to generate it, then re-run this script.'], outcome_perm_file);

else

    % Load into a struct so nothing in the results workspace is overwritten.
    OP = load(outcome_perm_file);

    assert(numel(OP.Y_labels) == n_outcomes, ...
        'Number of outcomes in %s (%d) does not match n_outcomes (%d).', ...
        outcome_perm_file, numel(OP.Y_labels), n_outcomes);

    % Null distribution summaries, from the raw permutation draws
    null_mean_o = mean(OP.null_dR2_outcomes, 1, 'omitnan');
    null_sd_o   = std(OP.null_dR2_outcomes, 0, 1, 'omitnan');
    null_ci_o   = prctile(OP.null_dR2_outcomes, [2.5, 97.5], 1);

    fprintf(['dR2 = R2(Threat/Dep+Cog) - R2(Threat/Dep)\n' ...
        'Observed: mean +/- SD over %d cross-validation repeats\n' ...
        'Null:     mean +/- SD over %d permutations\n' ...
        '(intervals are 2.5th - 97.5th percentiles of each distribution)\n'], ...
        size(OP.observed_dR2_outcomes, 1), OP.n_perms);

    for outcome_idx = 1:n_outcomes

        fprintf('\n%s\n', outcome_labels{outcome_idx});

        fprintf('  Observed dR2:  %9.5f +/- %.5f   95%% interval [%9.5f, %9.5f]\n', ...
            OP.observed_dR2_mean(outcome_idx), ...
            OP.observed_dR2_sd(outcome_idx), ...
            OP.observed_dR2_ci(1, outcome_idx), ...
            OP.observed_dR2_ci(2, outcome_idx));

        fprintf('  Null dR2:      %9.5f +/- %.5f   95%% interval [%9.5f, %9.5f]\n', ...
            null_mean_o(outcome_idx), ...
            null_sd_o(outcome_idx), ...
            null_ci_o(1, outcome_idx), ...
            null_ci_o(2, outcome_idx));

        fprintf('  Observed percentile within null distribution: %.1f%%\n', ...
            OP.percentile_outcomes(outcome_idx));

        fprintf('  p (one-tailed, H1: cognition adds value, uncorrected): %.4f\n', ...
            OP.p_uncorrected(outcome_idx));

        fprintf('  q (Benjamini-Hochberg FDR across %d outcomes):         %.4f\n', ...
            n_outcomes, OP.q_fdr(outcome_idx));

        if OP.significant_fdr(outcome_idx)
            fdr_result = 'YES';
        else
            fdr_result = 'NO';
        end

        fprintf('  Significant after FDR correction (q < 0.05): %s\n', fdr_result);
    end

    fprintf('\nOutcomes significant after FDR correction (q < 0.05): %d of %d\n', ...
        sum(OP.significant_fdr), n_outcomes);

    if any(OP.significant_fdr)
        fprintf('  %s\n', strjoin(outcome_labels(OP.significant_fdr), ', '));
    end

end