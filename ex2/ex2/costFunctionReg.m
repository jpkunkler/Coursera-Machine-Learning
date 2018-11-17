function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% exclude theta(1) because theta(1) is not to be regularized!
reg = (lambda/(2*m))*sum(theta(2:length(theta)).^2);

J = 1/m * sum(-y'*log(sigmoid(X*theta))-(1.-y)'*log(1-sigmoid(X*theta))) + reg;

% mask will be used such that the first theta value will be multiplied by 0, therefore will be excluded
% while the remaining theta values will be multiplied by 1 and therefore be regularized
mask = ones(size(theta));
mask(1) = 0;

% thanks to our mask, for j=0 our gradient will be identical to the normal gradient without regularization
grad =  1/m * ((sigmoid(X*theta)-y)'*X)' + lambda/m * (theta.*mask);
% =============================================================

end
