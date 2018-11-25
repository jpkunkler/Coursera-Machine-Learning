function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a1 = [ones(m,1) X];

z2 = a1 * Theta1';
a2 = [ones(size(z2,1),1) sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

% build our result vectors for every y-element! because 3 would be classified as [0 0 0 1 0 0 0 0 0 0] and so on.
% create identity matrix and index it by using y as row-index for the identity matrix
% for every y value (1 to 9), return 1 at exactly one position like depicted above for example value 3

%Octave short-hand version of the for-loop!
%Y = eye(num_labels)(y, :);

% the same result could also be achieved via longer for-loop!
Y = zeros(m,num_labels);
for i = 1:m
  for j = 1:num_labels
    % since Octave starts indexing at 1, we map our y-value of 0 to 10!
    % a y-value of 0 corresponds to [0 0 0 0 0 0 0 0 0 1]!
    if y(i) == 0
      label = 10;
    else
      label = y(i);
    endif
    % set our Y matrix of 10-dimensional result-vectors y for each result of X
    % to value of 1 at the label index that is true and 0 otherwise
    Y(i,label) = 1;
  endfor
endfor

% Compute Cost function w/o Regularization
cost = sum((Y .* log(a3)) + ((1 - Y) .* log(1 - a3)));
J = -(1 / m) * sum(cost);

% ------ REGULARIZATION ------
% Theta1 and Theta2 are set up horizontal like [theta0 theta1 theta2 theta3 ... thetan]
% so we start at the 2nd column because we do not include theta0 in regularization!
% We need all rows of Theta-Vectors because those map weights to all neurons (ai_0 to ai_j) in the layer
reg = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J +reg;

Delta1 = 0;
Delta2 = 0;

for t = 1:m
  % forward pass for t-th training example
  a1 = [1; X(t,:)'];
  z2 = Theta1 * a1;
  a2 = [1; sigmoid(z2)];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);

  % calculate delta values for backprop
  d3 = a3 - Y(t,:)';
  d2 = (Theta2(:, 2:end)')*d3.*sigmoidGradient(z2);

  % aggregate individual deltas
  Delta2 += d3*a2';
  Delta1 += d2*a1';
endfor

% Normalize
Theta1_grad = (1/m)*Delta1;
Theta2_grad = (1/m)*Delta2;

% add regularization term to all theta except for bias unit
Theta1_grad(:, 2:end) += ((lambda/m)*Theta1(:, 2:end));
Theta2_grad(:, 2:end) += ((lambda/m)*Theta2(:, 2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


endfunction
