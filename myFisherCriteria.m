function fisherCriteria = myFisherCriteria(featureMatrix, index_pos, index_neg)
    mean0 = mean(featureMatrix, 2);
    mean1 = mean(featureMatrix(:, index_pos), 2);
    mean2 = mean(featureMatrix(:, index_neg), 2);

    var1 = var(featureMatrix(:, index_pos), 0, 2);
    var2 = var(featureMatrix(:, index_neg), 0, 2);

    fisherCriteria = ((mean0 - mean1).^2 + (mean0 - mean2).^2) ./ (var1 + var2);
end
