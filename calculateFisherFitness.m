function fitness = calculateFisherFitness(selectedFeatures, featureMatrix)
    % Assume selectedFeatures is a binary vector indicating the selected features
    % Calculate Fisher fitness based on your provided criteria

    N1 = sum(selectedFeatures);
    N2 = sum(~selectedFeatures);

    mu1 = mean(featureMatrix(selectedFeatures, :));
    mu2 = mean(featureMatrix(~selectedFeatures, :));
    mu0 = mean(featureMatrix);

    s1 = cov(featureMatrix(selectedFeatures, :));
    s2 = cov(featureMatrix(~selectedFeatures, :));

    S_w = s1 + s2;
    S_b = (mu1 - mu0)' * (mu1 - mu0) + (mu2 - mu0)' * (mu2 - mu0);

    J = trace(S_b) / trace(S_w);

    % Invert J because genetic algorithm minimizes, and Fisher's criterion is maximized
    fitness = 1 / J;
end
