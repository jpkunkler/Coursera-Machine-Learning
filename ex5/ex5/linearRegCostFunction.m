function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% COST FUNCTION -----------------
reg = (lambda/(2*m))*sum(theta(2:length(theta)).^2);

J = 1/(2*m) * sum((X*theta - y).^2) + reg;

% GRADIENT --------------------

% mask will be used such that the first theta value will be multiplied by 0, therefore will be excluded
% while the remaining theta values will be multiplied by 1 and therefore be regularized
mask = ones(size(theta));
mask(1) = 0;

% thanks to our mask, for j=0 our gradient will be identical to the normal gradient without regularization
grad =  1/m * ((X*theta-y)'*X)' + lambda/m * (theta.*mask);











% =========================================================================

grad = grad(:);

end
