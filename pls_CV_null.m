function null_mdl = pls_CV_null(y,x,npls_comps, n_outer_reps,n_inner_reps )
% function for pls regression to generate a null model via shuffling at
% large n_outer_reps allows for significance testing of beta weights, at
% small only allows for significance testing of validation
%
% y - matrix of subject*outcomes that will be predicted by plsr
% x - matrix of subject*predictors that will be used to predict y
% npls_comps - max number of pls components explored in inner cv - note: it
% should be redundant to explore large numbers of components (i.e.
% approaching to npls_comps = MIN(SIZE(X,1)-1, SIZE(X,2)))
% n_outer_reps - number of outer CV repeats for model validation
% n_inner_reps - number of outer CV repeats for hyperparameter tuning

nSubs = length(y);
% nested crossvalidation for hyperparameter (NCOMPS) tuning
k_outer = 5;
k_inner = 3;

for i_outer = 1:n_outer_reps
    
    % define cv folds
    cv_indices_outer = crossvalind('Kfold', nSubs, k_outer); % on each repetition creates a new index
    y = y(randperm(length(y)),:); % permute observed data

    for j_outer = 1:k_outer % for each cross validation fold
        
        % divide training and validation folds
        test = (cv_indices_outer == j_outer);
        train = ~test;
        
        x_train = x(train,:);
        y_train = y(train,:);
        
        x_test = x(test,:);
        y_test = y(test,:);
        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% tuning - optimal number of pls components %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        n_pls_comps = 3;  % min(size(x,1) - 1,size(x,2));

        for i_inner = 1:n_inner_reps
            % define inner cv folds
            cv_indices_inner = crossvalind('Kfold', length(y_train), k_inner);
            for j_inner = 1:k_inner % for each cross validation fold

                % divide training and validation folds
                test_inner = (cv_indices_inner == j_inner);
                train_inner = ~test_inner;

                x_train_inner = x_train(train_inner,:);
                y_train_inner = y_train(train_inner,:);

                x_test_inner = x_train(test_inner,:);
                y_test_inner = y_train(test_inner,:);

                for comps = 1:n_pls_comps % predict unseen scores with each n_pls_comps

                    % fit pls regression with target
                    [~,~,~,~,BETA,~,~] = plsregress(x_train_inner, y_train_inner, comps);

                    % predict unseen sample data, BETA includes constant
                    ests_inner = [ones(size(x_test_inner,1),1) x_test_inner]*BETA;

                    SStot = sum((y_test_inner - mean(y_test_inner)).^2);
                    SSres = sum((y_test_inner - ests_inner).^2);
                    e = 1 - SSres./SStot;

                    % e = mean((y_test_inner - ests_inner).^2);% Mean Squared Error

                    err(j_inner,comps) = mean(e);

                end % end inner - number of comps to consider
                 err_cv= mean(err);
            end % end inner - CV folds
        end % end inner - CV repeats

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% fit the model using n-comps that are needed to produce minimum
%%%%%%%% error during validation of tuning sample %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [~,I] = max(mean(err_cv)); % find number of components with minimum error
        
        % fit pls model using that many components
        [~,~,~,~,BETA,~,~,~] = plsregress(x_train, y_train, I);
        
        % predict scores of validation fold
        ests = [ones(size(x_test,1),1) x_test]*BETA;
        err_outer= mean((y_test - ests).^2);
            

                    SStot = sum((y_test - mean(y_test)).^2);
                    SSres = sum((y_test - ests).^2);
                    acc = 1 - SSres./SStot;

            % acc(1,beh) = corr(y_test(:,beh), ests(:,beh));

        null_mdl.BETA{i_outer,j_outer} = BETA;
        null_mdl.mse_outer{i_outer,j_outer} = err_outer;
        null_mdl.R_outer{i_outer,j_outer} = acc;

    end
end

end
