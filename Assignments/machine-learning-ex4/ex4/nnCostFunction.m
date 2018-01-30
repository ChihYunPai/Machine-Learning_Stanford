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

% -------------------------------------------------------------
a1 = [ones(m,1),X]; % 5000 x 401
Y=(repmat(1:num_labels,m,1)==repmat(y,1,num_labels));
% X      : 5000 x 401
% Y      : 5000 x 10
% Theta1 : 25 x 401
% Theta2 : 10 x 26

% 1. Set the input layer's values to the t-th training example X(t,:). Perform a feedforward pass, computing the activations (z2,a2,z3,a3) for layer 2 and 3.
z2 = a1*Theta1'; % z2: 5000 x 25
a2 = [ones(m,1),sigmoid(z2)]; % a2 : 5000 x 26
z3 = a2*Theta2'; % z3: 5000 x 10
a3 = sigmoid(z3); % a3 : 5000 x 10
J = -mean(sum(Y.*log(a3) + (1-Y).*log(1-a3),2)) + (.5*lambda/m)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% 2. For each output unit k in layer3 (the output layer), set d3(k) = a3(k) - y(k)
d3 = (a3'-Y'); % d3: 10 x 5000
% 3. For the hidden layer l=2, set d2 = Theta2'*d3
d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z2'); % d2: 25 x 5000
% 4. Accumulate the gradient from this example using Delta_1 = Delta
% 5. Obtain the (unregularized) gradient for the neural network cost function by dividing the accumulated gradients by 1/m
Theta1_grad = (d2*(a1) + lambda*[zeros(size(Theta1,1),1),Theta1(:,2:end)])/m;
Theta2_grad = (d3*(a2) + lambda*[zeros(size(Theta2,1),1),Theta2(:,2:end)])/m;

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
