function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); % grad:[28x1]
h = sigmoid(X*theta);
reg_theta = [[0]; theta([2:length(theta)])]; % reg_theta:[28x1], only the 1st element is 0 without changing

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J =sum(log(h)'*-y-log(1-h)'*(1-y))*1/m + (lambda/(2*m))*sum(reg_theta.^2);

grad = (1/m)* sum((h-y).*X) + (lambda/m)*reg_theta'; % [1x28] + [1x28]

% =============================================================

end
