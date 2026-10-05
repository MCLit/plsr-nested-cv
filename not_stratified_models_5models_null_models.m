clear
clc

% Set up parallel processing
nslots = str2double(getenv('NSLOTS'));
parpool(nslots);
pp = gcp;
fprintf('Parpool size %d\n', pp.NumWorkers);

data = readtable('CTQ_DAWBA_CANTAB_FU2_complete.csv', 'VariableNamingRule', 'preserve');

%% Extract variables
Y_labels = {'FU2_specphobia', 'FU2_socphobia', 'FU2_agoraphobia', 'FU2_ocd', ...
    'FU2_anxiety', 'FU2_depression', 'FU2_ptsd','FU2_conductdisorder', 'FU2_hyperactivity'};
y = table2array(data(:, Y_labels));

age_data = data.FU2_age;
sex_numeric = strcmp(data.sex, 'M');
[~, ~, site_data] = unique(data.("recruitment.centre"));

%% Define predictors
X_trauma = [age_data, sex_numeric, table2array(data(:, {'CTQ_emotional_abuse', 'CTQ_physical_abuse', 'CTQ_sexual_abuse', 'CTQ_emotional_neglect', 'CTQ_physical_neglect'}))];
X_cognition = [age_data, sex_numeric, table2array(data(:, {'AGN_latency_pos', 'AGN_latency_neg', 'CGT_delay_aversion', 'CGT_deliberation', 'CGT_bet', 'CGT_quality', 'CGT_risk_adj', 'CGT_risk_taking', 'RPV_attention', 'SWM_errors', 'SWM_strategy'}))];
X_combined = [age_data, sex_numeric, table2array(data(:, {'CTQ_emotional_abuse', 'CTQ_physical_abuse', 'CTQ_sexual_abuse', 'CTQ_emotional_neglect', 'CTQ_physical_neglect', 'AGN_latency_pos', 'AGN_latency_neg', 'CGT_delay_aversion', 'CGT_deliberation', 'CGT_bet', 'CGT_quality', 'CGT_risk_adj', 'CGT_risk_taking', 'RPV_attention', 'SWM_errors', 'SWM_strategy'}))];
X_threat_dep = [age_data, sex_numeric, table2array(data(:, {'CTQ_threat', 'CTQ_deprivation'}))];
X_threat_dep_cog = [age_data, sex_numeric, table2array(data(:, {'CTQ_threat', 'CTQ_deprivation', 'AGN_latency_pos', 'AGN_latency_neg', 'CGT_delay_aversion', 'CGT_deliberation', 'CGT_bet', 'CGT_quality', 'CGT_risk_adj', 'CGT_risk_taking', 'RPV_attention', 'SWM_errors', 'SWM_strategy'}))];

%% Run null models (CV only)
all_null_models{1} = run_null_cv(X_trauma, y, site_data);
all_null_models{2} = run_null_cv(X_cognition, y, site_data);
all_null_models{3} = run_null_cv(X_combined, y, site_data);
all_null_models{4} = run_null_cv(X_threat_dep, y, site_data);
all_null_models{5} = run_null_cv(X_threat_dep_cog, y, site_data);

save('null_models_5models.mat', 'all_null_models', 'Y_labels');

function mdl = run_null_cv(X, y, site_data)
    % Remove missing data
    complete_rows = ~any(isnan([X, y, site_data]), 2);
    X = X(complete_rows, :);
    y = y(complete_rows, :);
    site_data = site_data(complete_rows);
    
    % Run null CV with shuffled y
    mdl.cv_results = null_pls_multivariate_nestedCV(X, y, site_data, 1000);
end

function mdl = null_pls_multivariate_nestedCV(X, y, confounds, n_repeats)
    n_subjects = size(X, 1);
    n_responses = size(y, 2);
    k_outer = 5;
    k_inner = 3;
    
    % Check if confounds are in cell format
    if iscell(confounds)
        confounds_numeric = cell2mat(confounds);
    else
        confounds_numeric = confounds;
    end
    confounds_numeric = double(confounds_numeric);
    
    n_preds = size(X, 2);
    max_components = min([n_preds, n_responses, 4]);
    
    % Initialize storage
    
    % Pre-allocate mdl structure
    mdl(n_repeats) = struct();
    for i = 1:n_repeats
        mdl(i).optimal_components = [];
        mdl(i).performance = struct('train', struct(), 'test', struct());
        mdl(i).data = struct('train', struct(), 'test', struct());
    end
    
    fprintf('Starting NULL PLS CV with %d repeats, %d outer folds\n', n_repeats, k_outer);
    parfor repeat = 1:n_repeats
        if mod(repeat, 100) == 0
            fprintf('[%s] Completed repeat %d/%d\n', datetime('now'), repeat, n_repeats);
        end
        
        % Initialize for this repeat
        subject_train_predictions = zeros(n_subjects, n_responses);
        subject_train_actuals = zeros(n_subjects, n_responses);
        train_predictions = [];
        train_actuals = [];
        fold_test_R2 = [];
        fold_test_R = [];
        subject_predictions = zeros(n_subjects, n_responses);
        subject_actuals = zeros(n_subjects, n_responses);
        optimal_comps_fold = zeros(k_outer, 1);
        
        % OUTER CV LOOP
        cv_indices = crossvalind('Kfold', n_subjects, k_outer);
        
        % SHUFFLE RESPONSES FOR NULL MODEL
        y_shuffled = y(randperm(size(y,1)), :);
        
        for fold = 1:k_outer
            % Get outer train/test split
            outer_test_idx = (cv_indices == fold);
            outer_train_idx = ~outer_test_idx;
            curr_test_idx = find(outer_test_idx);
            train_idx_nums = find(outer_train_idx);
            
            X_outer_train = X(outer_train_idx, :);
            X_outer_test = X(outer_test_idx, :);
            y_outer_train = y_shuffled(outer_train_idx, :);
            y_outer_test = y_shuffled(outer_test_idx, :);
            D_outer_train = confounds_numeric(outer_train_idx, :);
            D_outer_test = confounds_numeric(outer_test_idx, :);
            
            % INNER CV LOOP: Hyperparameter tuning
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
                        D_train_mean = mean(D_inner_train, 1);
                        D_train_std = std(D_inner_train, 1);
                        D_train_std(D_train_std < eps) = eps;
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
                        
                        % CONFOUND REGRESSION on normalized data
                        try
                            [B_X, ~] = mvregress(D_inner_train_norm, X_inner_train_norm);
                            X_inner_train_resid = X_inner_train_norm - D_inner_train_norm * B_X;
                            X_inner_test_resid = X_inner_test_norm - D_inner_test_norm * B_X;
                        catch
                            B_X = D_inner_train_norm \ X_inner_train_norm;
                            X_inner_train_resid = X_inner_train_norm - D_inner_train_norm * B_X;
                            X_inner_test_resid = X_inner_test_norm - D_inner_test_norm * B_X;
                        end

                        try
                            [B_Y, ~] = mvregress(D_inner_train_norm, y_inner_train_norm);
                            y_inner_train_resid = y_inner_train_norm - D_inner_train_norm * B_Y;
                            y_inner_test_resid = y_inner_test_norm - D_inner_test_norm * B_Y;
                        catch
                            B_Y = D_inner_train_norm \ y_inner_train_norm;
                            y_inner_train_resid = y_inner_train_norm - D_inner_train_norm * B_Y;
                            y_inner_test_resid = y_inner_test_norm - D_inner_test_norm * B_Y;
                        end

                        % PLS MODEL
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
            
            % Select optimal components
            [~, optimal_components] = max(component_performance);
            optimal_comps_fold(fold) = optimal_components;
            
            % FINAL MODEL TRAINING ON OUTER DATA
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
            
            % CONFOUND REGRESSION on normalized data
            try
                [B_X, ~] = mvregress(D_outer_train_norm, X_outer_train_norm);
                X_outer_train_resid = X_outer_train_norm - D_outer_train_norm * B_X;
                X_outer_test_resid = X_outer_test_norm - D_outer_test_norm * B_X;
            catch
                B_X = D_outer_train_norm \ X_outer_train_norm;
                X_outer_train_resid = X_outer_train_norm - D_outer_train_norm * B_X;
                X_outer_test_resid = X_outer_test_norm - D_outer_test_norm * B_X;
            end

            try
                [B_Y, ~] = mvregress(D_outer_train_norm, y_outer_train_norm);
                y_outer_train_resid = y_outer_train_norm - D_outer_train_norm * B_Y;
                y_outer_test_resid = y_outer_test_norm - D_outer_test_norm * B_Y;
            catch
                B_Y = D_outer_train_norm \ y_outer_train_norm;
                y_outer_train_resid = y_outer_train_norm - D_outer_train_norm * B_Y;
                y_outer_test_resid = y_outer_test_norm - D_outer_test_norm * B_Y;
            end

            % TRAIN FINAL PLS MODEL
            [~, ~, ~, ~, beta] = plsregress(X_outer_train_resid, y_outer_train_resid, optimal_components);
            
            % Predictions
            train_pred = [ones(size(X_outer_train_resid,1),1), X_outer_train_resid] * beta;
            train_predictions = [train_predictions; train_pred];
            train_actuals = [train_actuals; y_outer_train_resid];
            
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
            fold_R = corr(test_pred(:), y_outer_test_resid(:));
            
            fold_test_R2 = [fold_test_R2; fold_R2];
            fold_test_R = [fold_test_R; fold_R];
            
            % Store predictions
            subject_train_predictions(train_idx_nums, :) = train_pred;
            subject_train_actuals(train_idx_nums, :) = y_outer_train_resid;
            subject_predictions(curr_test_idx, :) = test_pred;
            subject_actuals(curr_test_idx, :) = y_outer_test_resid;
        end
        
        % Simple averages across folds
        test_R2_overall = mean(fold_test_R2, 'omitnan');
        test_R_overall = mean(fold_test_R, 'omitnan');
        
        % Calculate individual outcome R² from stored predictions
        test_R2_per_response = zeros(1, n_responses);
        for resp = 1:n_responses
            pred_col = subject_predictions(:, resp);
            actual_col = subject_actuals(:, resp);
            
            % Remove NaN values
            valid_idx = ~isnan(pred_col) & ~isnan(actual_col);
            pred_col = pred_col(valid_idx);
            actual_col = actual_col(valid_idx);
            
            if length(pred_col) > 0
                SS_res = sum((pred_col - actual_col).^2);
                SS_tot = sum((actual_col - mean(actual_col)).^2);
                if SS_tot > eps
                    test_R2_per_response(resp) = 1 - SS_res/SS_tot;
                else
                    test_R2_per_response(resp) = 0;
                end
            end
        end
        
        % Training metrics
        train_R_overall = corr(train_predictions(:), train_actuals(:));
        train_R2_overall = 1 - sum((train_predictions(:) - train_actuals(:)).^2) / ...
                                sum((train_actuals(:) - mean(train_actuals(:))).^2);
        
        % Store results for this repeat
        mdl(repeat).optimal_components = optimal_comps_fold;
        mdl(repeat).performance.train.overall_R = train_R_overall;
        mdl(repeat).performance.train.overall_R2 = train_R2_overall;
        mdl(repeat).performance.test.overall_R = test_R_overall;
        mdl(repeat).performance.test.overall_R2 = test_R2_overall;
        mdl(repeat).performance.test.R2_per_response = test_R2_per_response;
        mdl(repeat).performance.test.fold_test_R2 = fold_test_R2;
        mdl(repeat).performance.test.fold_test_R = fold_test_R;
        
        mdl(repeat).data.train.train_predictions = subject_train_predictions;
        mdl(repeat).data.train.train_actuals = subject_train_actuals;
        mdl(repeat).data.test.test_predictions = subject_predictions;
        mdl(repeat).data.test.test_actuals = subject_actuals;
    end
    
    fprintf('NULL PLS CV complete\n');
end

function regressed_data = regress_site_effects(data, site_data)
    n_sites = length(unique(site_data));
    site_dummies = zeros(length(site_data), n_sites-1);
    for i = 1:n_sites-1
        site_dummies(:,i) = (site_data == i);
    end
    regressed_data = data - site_dummies * (site_dummies \ data);
end