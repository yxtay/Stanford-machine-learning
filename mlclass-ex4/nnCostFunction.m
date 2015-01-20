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

% values for hidden layer
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);

% values for output later
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
pred = a3;

% calculate unregularised cost function
% for k = 1:num_labels
%     J = J + (-(y == k)' * log(pred(:,k)) - (1 - (y == k)') * log (1 - pred(:,k)));
% end
y_mat = repmat(y, 1, num_labels) == repmat(1:num_labels, m, 1);
J = sum(sum(- y_mat .* log(pred) - (1 - y_mat) .* log(1 - pred), 2));
J = J / m;

% add regularisation
Theta1n0 = Theta1; Theta1n0(:,1) = 0;
Theta2n0 = Theta2; Theta2n0(:,1) = 0;
Thetan0 = [Theta1n0(:); Theta2n0(:)];
J = J + lambda / (2 * m) * (Thetan0' * Thetan0);

% backpropagation for gradient
% Delta2 = 0; Delta1 = 0;
% for t = 1:m
%     delta3 = (a3(t,:))' - (y_mat(t,:))';
%     delta2 = (Theta2(:,2:end))' * delta3 .* sigmoidGradient((z2(t,:))');
%     Delta2 = Delta2 + delta3 * a2(t,:);
%     Delta1 = Delta1 + delta2 * a1(t,:);
% end
delta3 = a3 - y_mat;
delta2 = delta3 * Theta2(:,2:end) .* sigmoidGradient(z2);
Delta2 = delta3' * a2;
Delta1 = delta2' * a1;

% gradient with regularisation
Theta2_grad = Delta2 / m + lambda / m * Theta2n0;
Theta1_grad = Delta1 / m + lambda / m * Theta1n0;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
