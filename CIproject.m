%%
% Ali KhosraviPour
clc;
clear all;
load("Project_data.mat");
[numberOfChannels, numberOfSamples, numberOfTrainTrials] = size(TrainData);
numberOfTestTrials = size(TestData, 3);
N_pos = length(find(TrainLabels == 1)); % positive emotions
N_neg = length(find(TrainLabels == -1)); % negative emotions
index_pos = find(TrainLabels == 1); % indexes of pos
index_neg = find(TrainLabels == -1); % indexes of neg

%% Phase 1:
%% Feature 1: Variance
featureName = 'Variance';
featureNumber = 1;
variance = zeros(numberOfChannels, numberOfTrainTrials);

% Calculate variance for each channel and trial
for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));
    variance(channel, :) = var(chData, 0, 2);
end

% Fisher Criteria:
fishers_var = myFisherCriteria(variance, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_var)), ' --> Channel Num = ', num2str(find(fishers_var == max(fishers_var)))])

%% Feature 2: Form Factor
featureName = 'Form Factor';
featureNumber = 2;
ff_values = zeros(numberOfChannels, numberOfTrainTrials);

% Calculate Form Factor for each channel and trial
for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    % Calculation of Form Factor for each trial
    for i = 1:numberOfTrainTrials
        first_derivative = diff(chData(i, :));
        second_derivative = diff(first_derivative);
        ff_value = (var(second_derivative) / var(first_derivative)) / (var(first_derivative) / var(chData(i, :)));
        ff_values(channel, i) = ff_value;
    end
end

% Fisher Criteria:
fishers_ff = myFisherCriteria(ff_values, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_ff)), ' --> Channel Num = ', num2str(find(fishers_ff == max(fishers_ff)))])

%% Feature 3: Mean Freq
featureName = 'Mean Freq';
mean_freqs = zeros(numberOfChannels, numberOfTrainTrials);
featureNumber = 3;

% Calculate Mean Freq for each channel and trial
for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    % Calculation of Mean Freq for each trial
    for i = 1:numberOfTrainTrials
        mean_freqs(channel, i) = meanfreq(chData(i, :));
    end
end

% Fisher Criteria:
fishers_meanFreq = myFisherCriteria(mean_freqs, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_meanFreq)), ' --> Channel Num = ', num2str(find(fishers_meanFreq == max(fishers_meanFreq)))])

%% Feature 4: Median Freq
featureName = 'Median Freq';
featureNumber = 4;
median_freqs = zeros(numberOfChannels, numberOfTrainTrials);

% Calculate Median Freq for each channel and trial
for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    % Calculation of Median Freq for each trial
    for i = 1:numberOfTrainTrials
        median_freqs(channel, i) = medfreq(chData(i, :));
    end
end

% Fisher Criteria:
fishers_medFreq = myFisherCriteria(median_freqs, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_medFreq)), ' --> Channel Num = ', num2str(find(fishers_medFreq == max(fishers_medFreq)))])

%% Feature 5: AR Coefs
featureName = 'AR Coefs';
featureNumber = 5;
order = 10;
ARcoefs_mean_values = zeros(numberOfChannels, numberOfTrainTrials);

% Calculate AR Coefs for each channel and trial
for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    % Calculation of AR Coefs and mean values for each trial
    for i = 1:numberOfTrainTrials
        ar_coefs = aryule(chData(i, :), order);
        ARcoefs_mean_values(channel, i) = mean(ar_coefs);
    end
end

% Fisher Criteria:
fishers_ARcoefs_mean = myFisherCriteria(ARcoefs_mean_values, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_ARcoefs_mean)), ' --> Channel Num = ', num2str(find(fishers_ARcoefs_mean == max(fishers_ARcoefs_mean)))])

%% Feature 6: Occupied Bandwidth
featureName = 'OBW';
featureNumber = 6;
obw_values = zeros(numberOfChannels, numberOfTrainTrials);

% Calculate OBW values for each channel and trial
for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    % Calculation of OBW values for each trial
    for i = 1:numberOfTrainTrials
        obw_values(channel, i) = obw(chData(i, :));
    end
end

% Fisher Criteria:
fishers_obw = myFisherCriteria(obw_values, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_obw)), ' --> Channel Num = ', num2str(find(fishers_obw == max(fishers_obw)))])


%% Feature 7: Band Power
featureName = 'Band Power';
featureNumber = 7;
bp_values = zeros(numberOfChannels, numberOfTrainTrials);

% Calculate Band Power values for each channel and trial
for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    % Calculation of Band Power values for each trial
    for i = 1:numberOfTrainTrials
        bp_values(channel, i) = bandpower(chData(i, :));
    end
end

% Fisher Criteria:
fishers_bp = myFisherCriteria(bp_values, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_bp)), ' --> Channel Num = ', num2str(find(fishers_bp == max(fishers_bp)))])

%% Feature 8: Maximum Power Frequency
featureName = 'Max Power';
featureNumber = 8;
mp_values = zeros(numberOfChannels, numberOfTrainTrials);

% Calculate Max Power values for each channel and trial
for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    % Calculation of Max Power values for each trial
    for i = 1:numberOfTrainTrials
        signal_len = length(chData(i, :));
        y = fftshift(fft(chData(i, :)));
        f = (-signal_len/2 : signal_len/2 - 1) * (fs / signal_len);      
        power = abs(y).^2 / signal_len;           
        [~, index] = max(power);
        mp_values(channel, i) = f(index);
    end
end

% Fisher Criteria:
fishers_mp = myFisherCriteria(mp_values, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_mp)), ' --> Channel Num = ', num2str(find(fishers_mp == max(fishers_mp)))])

%% Feature 9: Skewness
% Feature: Skewness
featureName = 'Skewness';
featureNumber = 9;
skewness_vals = zeros(numberOfChannels, numberOfTrainTrials);

% Calculate skewness values for each channel and trial
for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    % Calculation of skewness for each trial
    for i = 1:numberOfTrainTrials
        skewness_vals(channel, i) = skewness(chData(i, :));
    end
end

% Fisher Criteria for Skewness:
fishers_skewness = myFisherCriteria(skewness_vals, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_skewness)), ' --> Channel Num = ', num2str(find(fishers_skewness == max(fishers_skewness)))])

%% Feature 10: Average Power
featureName = 'Average Power';
featureNumber = 10;
average_power_vals = zeros(numberOfChannels, numberOfTrainTrials);

% Calculate average power values for each channel and trial
for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    % Calculation of average power for each trial
    for i = 1:numberOfTrainTrials
        average_power_vals(channel, i) = mean(abs(chData(i, :)).^2);
    end
end

% Fisher Criteria for Average Power:
fishers_average_power = myFisherCriteria(average_power_vals, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_average_power)), ' --> Channel Num = ', num2str(find(fishers_average_power == max(fishers_average_power)))])


%% Feature 11: Kurtosis
featureName = 'Kurtosis';
featureNumber = 11;
kurtosis_vals = zeros(numberOfChannels, numberOfTrainTrials);

for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    for i = 1:numberOfTrainTrials
        kurtosis_vals(channel, i) = kurtosis(chData(i, :));
    end
end

fishers_kurtosis = myFisherCriteria(kurtosis_vals, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_kurtosis)), ' --> Channel Num = ', num2str(find(fishers_kurtosis == max(fishers_kurtosis)))])

%% Feature 12: STD
featureName = 'Standard Deviation';
featureNumber = 12;
std_deviation_vals = zeros(numberOfChannels, numberOfTrainTrials);

for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    for i = 1:numberOfTrainTrials
        std_deviation_vals(channel, i) = std(chData(i, :));
    end
end

fishers_std_deviation = myFisherCriteria(std_deviation_vals, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_std_deviation)), ' --> Channel Num = ', num2str(find(fishers_std_deviation == max(fishers_std_deviation)))])

%% Feature 13: Crest Factor
featureName = 'Crest Factor';
featureNumber = 13;
crest_factor_vals = zeros(numberOfChannels, numberOfTrainTrials);

for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    for i = 1:numberOfTrainTrials
        crest_factor_vals(channel, i) = max(abs(chData(i, :))) / rms(chData(i, :));
    end
end

fishers_crest_factor = myFisherCriteria(crest_factor_vals, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_crest_factor)), ' --> Channel Num = ', num2str(find(fishers_crest_factor == max(fishers_crest_factor)))])


%% Feature 14: Zero Crossing Rate
featureName = 'Zero Crossing Rate';
featureNumber = 14;
zero_crossing_rate_vals = zeros(numberOfChannels, numberOfTrainTrials);

for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    for i = 1:numberOfTrainTrials
        zero_crossing_rate_vals(channel, i) = sum(abs(diff(sign(chData(i, :)))))/2;
    end
end

fishers_zero_crossing_rate = myFisherCriteria(zero_crossing_rate_vals, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_zero_crossing_rate)), ' --> Channel Num = ', num2str(find(fishers_zero_crossing_rate == max(fishers_zero_crossing_rate)))])

%% Feature 15: Spectral Entropy
featureName = 'Spectral Entropy';
featureNumber = 15;
spectral_entropy_vals = zeros(numberOfChannels, numberOfTrainTrials);

for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    for i = 1:numberOfTrainTrials
        [pxx, f] = pwelch(chData(i, :), [], [], [], fs);
        spectral_entropy_vals(channel, i) = -sum(pxx .* log2(pxx));
    end
end

fishers_spectral_entropy = myFisherCriteria(spectral_entropy_vals, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_spectral_entropy)), ' --> Channel Num = ', num2str(find(fishers_spectral_entropy == max(fishers_spectral_entropy)))])

%% Feature 16: Dominant Frequency
featureName = 'Dominant Frequency';
featureNumber = 16;
dominant_frequency_vals = zeros(numberOfChannels, numberOfTrainTrials);

for channel = 1:numberOfChannels
    chData = TrainData(channel, 1001:end, :);
    chData = transpose(squeeze(chData));

    for i = 1:numberOfTrainTrials
        [pxx, f] = pwelch(chData(i, :), [], [], [], fs);
        [~, index] = max(pxx);
        dominant_frequency_vals(channel, i) = f(index);
    end
end

fishers_dominant_frequency = myFisherCriteria(dominant_frequency_vals, index_pos, index_neg);
disp(['Feature ', num2str(featureNumber), ': ', featureName, ' --> Highest Fisher Score = ', num2str(max(fishers_dominant_frequency)), ' --> Channel Num = ', num2str(find(fishers_dominant_frequency == max(fishers_dominant_frequency)))])


%% Calculating Correlation Between 59 Channels

average_values = zeros(numberOfChannels, numberOfTrainTrials);
for channel = 1: numberOfChannels
    chData = TrainData(channel, 1001 : end , :);
    chData = transpose(squeeze(chData));
    for i = 1: numberOfTrainTrials
        average_values(channel, i) = sum(chData(i, :)) / 4000;
    end
end

correlation_between_channels = corr(average_values');
correlation_between_channels(logical(eye(size(correlation_between_channels)))) = 0;
correlation_threshold = 0.8;
high_correlation_indices = find(correlation_between_channels > correlation_threshold);
[row_indices, col_indices] = ind2sub(size(correlation_between_channels), high_correlation_indices);

disp('High Correlation Channels:');
disp([row_indices, col_indices]);

%% Concatenating Criteria Matrices
concatenated_matrix = vertcat(variance, ff_values, mean_freqs, median_freqs, ARcoefs_mean_values, obw_values, bp_values, mp_values, skewness_vals, average_power_vals, kurtosis_vals, std_deviation_vals, crest_factor_vals, zero_crossing_rate_vals, spectral_entropy_vals, dominant_frequency_vals);

%% Feature Selection Based on Fisher Score
fisher_scores = vertcat(fishers_var, fishers_ff, fishers_meanFreq, fishers_medFreq, fishers_ARcoefs_mean, fishers_obw, fishers_bp, fishers_mp, fishers_skewness, fishers_average_power);
[sorted_scores, sorted_indices] = sort(fisher_scores, 'descend');
top_250_rows = concatenated_matrix(sorted_indices(1:250), :);
top_250_rows = zscore(top_250_rows);
disp('Train Features Selected Successfully!')

%% Features For Test Data
% 1
variance_test = zeros(numberOfChannels, numberOfTestTrials);
% 2
ff_values_test = zeros(numberOfChannels, numberOfTestTrials);
first_derivative = zeros(1, 4000);
second_derivative = zeros(1, 4000);
% 3
mean_freqs_test = zeros(numberOfChannels, numberOfTestTrials);
% 4
median_freqs_test = zeros(numberOfChannels, numberOfTestTrials);
% 5
ARcoefs_mean_values_test = zeros(numberOfChannels, numberOfTestTrials);
% ARcoefs_cell_test = cell(numberOfChannels, numberOfTestTrials);
% 6
obw_values_test = zeros(numberOfChannels, numberOfTestTrials);
% 7
bp_values_test = zeros(numberOfChannels, numberOfTestTrials);
% 8
mp_values_test = zeros(numberOfChannels, numberOfTestTrials);
% 9
skewness_vals_test = zeros(numberOfChannels, numberOfTestTrials);
% 10
average_power_vals_test = zeros(numberOfChannels, numberOfTestTrials);
% 11
kurtosis_vals_test = zeros(numberOfChannels, numberOfTestTrials);
% 12
std_deviation_vals_test = zeros(numberOfChannels, numberOfTestTrials);
% 13
crest_factor_vals_test = zeros(numberOfChannels, numberOfTestTrials);
% 14
zero_crossing_rate_vals_test = zeros(numberOfChannels, numberOfTestTrials);
% 15
entropy_vals_test = zeros(numberOfChannels, numberOfTestTrials);
% 16
dominant_frequency_vals_test = zeros(numberOfChannels, numberOfTestTrials);



for channel = 1: numberOfChannels

    chData = TestData(channel, 1001 : end , :);
    chData = transpose(squeeze(chData));

    for i = 1: numberOfTestTrials
        variance_test(channel, i) = var(chData(i, :));
        %
        first_derivative = diff(chData(i, :));
        second_derivative = diff(first_derivative);
        ff_value = (var(second_derivative)/ var(first_derivative)) / (var(first_derivative)/ var(chData(i, :))) ;
        ff_values_test(channel, i) = ff_value;
        %
        mean_freqs_test(channel, i) = meanfreq(chData(i, :));
        %
        median_freqs_test(channel, i) = medfreq(chData(i, :));
        %
        ar_coefs = aryule(chData(i, :), order);
        ARcoefs_mean_values_test(channel, i) = mean(ar_coefs);
        %
        obw_values_test(channel, i) = obw(chData(i, :));
        %
        bp_values_test(channel, i) = bandpower(chData(i, :));
        %
        n = length(chData(i, :));
        y = fftshift(fft(chData(i, :)));
        f = (-n/2:n/2-1)*(fs/n);       % 0-centered frequency range
        power = abs(y).^2/n;           % 0-centered power
        index = find(power == max(power));
        mp_values_test(channel, i) = index(end);
        %
        skewness_vals_test(channel, i) = skewness(chData(i, :));
        %
        average_power_vals_test(channel, i) = mean(abs(chData(i, :)).^2);
        %
        kurtosis_vals_test(channel, i) = kurtosis(chData(i, :));
        %
        std_deviation_vals_test(channel, i) = std(chData(i, :));
        %
        crest_factor_vals_test(channel, i) = max(abs(chData(i, :))) / rms(chData(i, :));
        %
        zero_crossing_rate_vals_test(channel, i) = sum(abs(diff(sign(chData(i, :)))) / 2);
        %
        entropy_vals_test(channel, i) = entropy(chData(i, :));
        %
        [pxx, f] = pwelch(chData(i, :), [], [], [], fs);
        [~, index] = max(pxx);
        dominant_frequency_vals_test(channel, i) = f(index);
    end
end

%% Concatenating Test Features
concatenated_matrix_test = vertcat(variance_test, ff_values_test, mean_freqs_test, median_freqs_test, ARcoefs_mean_values_test, obw_values_test, bp_values_test, mp_values_test, skewness_vals_test, average_power_vals_test, kurtosis_vals_test, std_deviation_vals_test, crest_factor_vals_test, zero_crossing_rate_vals_test, entropy_vals_test, dominant_frequency_vals_test);
featureMatrix_test = concatenated_matrix_test(sorted_indices(1:250), :);
featureMatrix_test = zscore(featureMatrix_test);
disp('Test Features Selected Successfully!')

%% MLP Training and Cross-Validation
disp('-------------------------------------------------------------------')
disp('MLP Networks: ')
% Selectable activation functions
activation_functions = ["tansig", "hardlims", "purelin"];
best_accuracy = -inf;
optimal_hidden_neurons = 0;

for activation_func = activation_functions
    for hidden_neurons = 1:20
        accuracies = zeros(1, 5);
        for fold = 1:5
            % 5-fold cross-validation
            train_indices = [1 : (fold - 1) * 110, fold * 110 + 1 : 550];
            valid_indices = (fold - 1) * 110 + 1 : fold * 110;
            % Splitting Data into Train & Validation
            TrainX = top_250_rows(:, train_indices);
            ValX = top_250_rows(:, valid_indices);
            TrainY = TrainLabels(train_indices);
            ValY = TrainLabels(valid_indices);
            % Training network
            net = patternnet(hidden_neurons);
            % net.trainParam.epochs = 2000;
            net = train(net, TrainX, TrainY);
            net.layers{2}.transferFcn = activation_func;
            % Predicting labels
            predict_y = net(ValX);
            predict_y(predict_y < 0) = -1;
            predict_y(predict_y >= 0) = 1;
            % Calculation of the accuracies
            % accuracies(fold) = sum(predict_y == ValY) / length(ValY);
            % if accuracies(fold) > best_accuracy
            %     best_accuracy = accuracies(fold);
            %     optimal_hidden_neurons = hidden_neurons;
            % end
        end
        
        avg_accuracy = mean(accuracies);
        if avg_accuracy > best_accuracy
            best_accuracy = avg_accuracy;
            optimal_hidden_neurons = hidden_neurons;
        end
        
    end
    
    disp(['Activation Func: ', activation_func, ' Num H-Neurons: ', num2str(optimal_hidden_neurons), ' Accuracy: ', num2str(best_accuracy * 100)]);
end

%% MLP Network for Test
% best values from previous section
best_activation_function = "purelin";
best_hidden_neurons = 20;

% Train MLP network with the best parameters on the entire training set
final_net = patternnet(best_hidden_neurons);
final_net.layers{2}.transferFcn = best_activation_function;
final_net = train(final_net, top_250_rows, TrainLabels);

test_predictions_MLP1 = final_net(featureMatrix_test);
test_predictions_MLP1(test_predictions_MLP1 < 0) = -1;
test_predictions_MLP1(test_predictions_MLP1 >= 0) = 1;

save('test_predictions_MLP1.mat', 'test_predictions_MLP1');


%% RBF Network Training and Cross-Validation
disp('-------------------------------------------------------------------')
disp('RBF Networks: ')

% Parameters
hidden_neurons_range = 1:20;
sigma_range = 1:0.1:4;

best_accuracy = -inf;
optimal_hidden_neurons = 0;
optimal_sigma = 0;

for hidden_neurons = hidden_neurons_range
    for sigma = sigma_range
        % 5-fold cross-validation
        accuracies = zeros(1, 5);
        
        for fold = 1:5
            train_indices = [ 1 : (fold - 1) * 110, fold * 110 + 1 : 550];
            valid_indices = (fold - 1) * 110 + 1 : fold * 110;
            % Cross-validation on RBF network
            TrainX = top_250_rows(:, train_indices);
            ValX = top_250_rows(:, valid_indices);
            TrainY = TrainLabels(train_indices);
            ValY = TrainLabels(valid_indices);
            net = newrb(TrainX, TrainY, 0, sigma, hidden_neurons, 5);
            % Predicting
            predict_y = sim(net, ValX);
            predict_y(predict_y < 0) = -1;
            predict_y(predict_y >= 0) = 1;
            % Calcualting Accuracy
            accuracies(fold) = sum(predict_y == ValY) / length(ValY);
            if accuracies(fold) > best_accuracy
                best_accuracy = accuracies(fold);
                optimal_hidden_neurons = hidden_neurons;
                optimal_sigma = sigma;
            end
        end
        % avg_accuracy = mean(accuracies);
        % if avg_accuracy > best_accuracy
        %     best_accuracy = avg_accuracy;
        %     optimal_hidden_neurons = hidden_neurons;
        %     optimal_sigma = sigma;
        % end  
    end
    disp(['Hidden Neurons: ', num2str(optimal_hidden_neurons), ' | Sigma: ', num2str(optimal_sigma), ' | Accuracy: ', num2str(best_accuracy * 100)]);
end


%% RBF Network for Test
% best values from previous section
optimal_hidden_neurons_rbf = 15;
optimal_sigma_rbf = 3;
final_rbf_net = newrb(top_250_rows, TrainLabels, 0, optimal_sigma_rbf, optimal_hidden_neurons_rbf, 5);

test_predictions_RBF1 = sim(final_rbf_net, featureMatrix_test);
test_predictions_RBF1(test_predictions_RBF1 < 0) = -1;
test_predictions_RBF1(test_predictions_RBF1 >= 0) = 1;

save('test_predictions_RBF1.mat', 'test_predictions_RBF1');

%% Phase 2: Using Genetic 
% Coding
%% Tournament Selection Function


% Main Genetic Algorithm Code
% Parameters
populationSize = n;
numFeatures = 250;
selectedFeaturesCount = 70;
mutationRate = 0.01;
crossoverRate = 0.8;
generations = 100;
tournamentSize = 5; % Adjust the tournament size as needed

% Initialize population
population = randi([0, 1], [populationSize, numFeatures]);

for generation = 1:generations
    % Evaluate fitness
    fitness = zeros(populationSize, 1);
    for i = 1:populationSize
        selectedFeatures = find(population(i, :));
        % Calculate Fisher criteria fitness
        fitness(i) = calculateFisherFitness(selectedFeatures, top_250_rows);
    end
    
    % Selection
    selectedIndices = tournamentSelection(population, fitness, tournamentSize);

    % Crossover
    population = crossover(population(selectedIndices, :), crossoverRate);

    % Mutation
    population = mutate(population, mutationRate);

    % Replace old generation with new one
    population(selectedIndices, :) = population;
end

% Find the best chromosome
bestChromosome = population(find(fitness == max(fitness)), :);
selectedFeatures = find(bestChromosome);

% Display the selected features
disp('Selected Features:');
disp(selectedFeatures);

function selectedIndices = tournamentSelection(population, fitness, tournamentSize)
    populationSize = size(population, 1);
    selectedIndices = zeros(populationSize, 1);

    for i = 1:populationSize
        % Randomly select indices for the tournament
        tournamentIndices = randi(populationSize, tournamentSize, 1);
        tournamentFitness = fitness(tournamentIndices);
        
        % Select the index with the highest fitness from the tournament
        [~, winnerIndex] = max(tournamentFitness);
        selectedIndices(i) = tournamentIndices(winnerIndex);
    end
end

% Crossover Function (Row Swapping)
function newPopulation = crossover(parents, crossoverRate)
    [numParents, ~] = size(parents);
    newPopulation = parents;

    % Iterate through pairs of parents for crossover
    for i = 1:2:numParents
        if rand() < crossoverRate
            % Randomly select two indices to swap rows
            indicesToSwap = randperm(size(parents, 2), 2);

            % Swap rows between two parents
            newPopulation(i, indicesToSwap) = parents(i+1, indicesToSwap);
            newPopulation(i+1, indicesToSwap) = parents(i, indicesToSwap);
        end
    end
end

% Mutation Function (Standard Mutation)
function newPopulation = mutate(population, mutationRate)
    [numIndividuals, numAlleles] = size(population);
    mutationMask = rand(numIndividuals, numAlleles) < mutationRate;
    mutationValues = rand(numIndividuals, numAlleles);
    newPopulation = population;

    % Apply mutation
    newPopulation(mutationMask) = mutationValues(mutationMask);
end

% ... (Rest of the code remains unchanged)


