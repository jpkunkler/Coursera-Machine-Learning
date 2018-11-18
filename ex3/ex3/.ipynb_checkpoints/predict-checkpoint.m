function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add column of 1s to our inputs as paramter x0
a1 = [ones(m,1) X];

% multiply input paramters by weights, add column a0 and calculate sigmoid hypothesis
z2 = a1 * Theta1';
a2 = [ones(size(z2),1) sigmoid(z2)];

% multiply first hidden layer by weights for hidden layer and calculate sigmoid hypothesis
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% retrieve the maximum value for each of the 9 possible outcomes --> most likely prediction
[max_values indices] = max(a3, [], 2);
p = indices;








% =========================================================================


end