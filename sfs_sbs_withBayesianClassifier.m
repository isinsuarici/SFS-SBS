

clear all;
close all;
clc;

% Load data
load("BIM488_HW3.mat")

% Split as train and test
training_1 = classes(1:80,:);
testing_1 = classes(81:160,:);
training_2 = classes(161:240,:);
testing_2 = classes(241:320,:);
training_3 = classes(321:400,:);
testing_3 = classes(401:480,:);
training_4 = classes(481:560,:);
testing_4 = classes(561:640,:);

% Merge train and test data.
train_set = {training_1, training_2, training_3, training_4};
test_set = {testing_1, testing_2, testing_3, testing_4};

% Calculate mean and cov for bayesian classification.
means= {(mean(training_1))',(mean(training_2))',(mean(training_3))',(mean(training_4))'};
covariance = {eye(8), eye(8), eye(8), eye(8)};
%covariance = {cov(training_1),cov(training_2),cov(training_3),cov(training_4)};

% Print results of sfs and sbs.
sfs_res = sfs(means, covariance, test_set);
disp("SFS")
disp("Optimum feature subset: ")
disp(sfs_res)
disp("Classification accuracy: ")
disp(classification_acccuracy(means, covariance, test_set, sfs_res))

sbs_res = sbs(means, covariance, test_set);
disp("SBS")
disp("Optimum feature subset: ")
disp(sbs_res)
disp("Classification accuracy: ")
disp(classification_acccuracy(means, covariance, test_set, sbs_res))


function selected_features = sfs(means, covariance, test_set)
    selected_features = [];
    accuracy = 0;
    all_features = 1:8;
    % Loop until adding a feature no longer improves the accuracy.
    while true
        highest_class = 0;
        highest_accuracy = accuracy;
        for i = setdiff(all_features, selected_features)
            test_features = [selected_features, i];
            new_accuracy = classification_acccuracy(means, covariance, test_set, test_features);
            if new_accuracy > highest_accuracy
                highest_accuracy = new_accuracy;
                highest_class = i;
            end
        end
        if highest_accuracy <= accuracy
            break;
        end
        accuracy = highest_accuracy;
        selected_features = [selected_features, highest_class];
    end
end


function selected_features = sbs(means, covariance, test_set)
    selected_features = 1:8;
    accuracy = classification_acccuracy(means, covariance, test_set, selected_features);
    % Loop until removing a feature no longer improves the accuracy.
    while true
        deleted_class = 0;
        hightest_accuracy = accuracy;
        for i = selected_features
            test_features = selected_features(i ~= selected_features);
            new_accuracy = classification_acccuracy(means, covariance, test_set, test_features);
            if new_accuracy >= hightest_accuracy
                hightest_accuracy = new_accuracy;
                deleted_class = i;
            end
        end

        if deleted_class==0
            % If this condition is met, 
            % an option that increases (or equals) accuracy was not found.
            break;
        end
        accuracy = hightest_accuracy;
        selected_features = selected_features(deleted_class ~= selected_features);
    end
end

function accuracy = classification_acccuracy(means, covariance, test_set, test_features)
    % Find accuracy with selected feature(s) in test set.
    corrects = 0;
            for class = 1:4
                for vector_num = 1:80
                    vec = zeros(1,8);
                    vec(test_features) = 1;
                    vec = vec .* test_set{class}(vector_num, :);
                    predicted = BayesianClassifier(means, covariance, vec');
                    if predicted == class
                        corrects = corrects + 1;
                    end
                end
            end
    accuracy = corrects/320*100;
end


function classification_res = BayesianClassifier(mean,covariance, x)
    % Find class for a data. 
    % For which class the data has higher bayes value, it will belong to
    % that class.(classification_res)
    max=0;
    for i=1:4
        bayes = 1/4 * comp_gauss_dens_val(mean{i},covariance{i},x);
        if bayes > max
            max = bayes;
            classification_res = i;
        end
    end
end


function pg = comp_gauss_dens_val(m, S, x) % m = mean vector, S = covariance matrix
    [l, c]=size(m);
    pg = (1/( (2*pi)^(l/2)*det(S)^0.5) )*exp(-0.5*(x-m)'*inv(S)*(x-m));
end

