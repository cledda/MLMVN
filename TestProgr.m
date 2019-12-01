Learn = load('arithm_prog_learn.txt');
Test = load('arithm_prog_test.txt');

Results = MLMVN('sizeOfMlmvn', [2 1], 'inputs', Learn, 'stoppingCriteria', 'rmse', 'discreteInput', 1, 'discreteOutput', 1, 'globalthresholdvalue', 0, 'localThresholdValue', 0, 'SoftMargins', 1, 'angularGlobalThresholdValue', 0.001, 'angularLocalThresholdValue', 0,'initialWeights','random', 'numberOfSectors', 64, 'maxIterations', 100);
Weights = Results.network;
Prediction = MLMVN('network', Weights, 'inputs', Test, 'stoppingCriteria', 'test', 'discreteInput', 1, 'discreteOutput', 1, 'globalthresholdvalue', 0.1, 'localThresholdValue', 0, 'numberOfSectors', 64);
disp('Desired Outputs');
disp(Prediction.DesiredOutputs);
disp('Actual Outputs');
disp(Prediction.NetworkOutputs);
figure(1);
hold off
plot(Prediction.DesiredOutputs, 'or');
hold on
plot(Prediction.NetworkOutputs, '*b');