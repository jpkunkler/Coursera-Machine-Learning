function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in
%               error_train and the validation errors in error_val. The
%               vector lambda_vec contains the different lambda parameters
%               to use for each calculation of the errors, i.e,
%               error_train(i), and error_val(i) should give
%               you the errors obtained after training with
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%
%       end
%
%

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    theta = trainLinearReg(X,y,lambda); % optimize theta for every single lambda-value

    % important: set lambda to 0 because we do not want regularization for our cost function J_train & J_cv
    % Why? See https://stats.stackexchange.com/q/222535
    % basically, we already used regularization in optimizing theta
    % so we can now set lambda = 0 to set regularization aside
    % to compare Train and CV-error on "even ground"
    %--> because lambda is already taken care off by our trainLinearReg()
    error_train(i) = linearRegCostFunction(X,y,theta,0);
    error_val(i) = linearRegCostFunction(Xval,yval,theta,0);

    % regularization is only needed while optimizing our model to account for overfitting training data
    % It is used to penalize too many features that have a small impact on our prediction
    % up to a point, this drastically reduces variance (aka flexibility) without increasing bias
    % --> helps our model to better be able to generalize because flexibility is reduced to an optimum

end





% =========================================================================

end
