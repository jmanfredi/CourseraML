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

%% Make predictions
hTheta = X*theta;
%%Compute cost function with no regularization using square error
JNoReg = (1/(2*m))*sum((hTheta - y).^2);
%% Element wise squaring of theta
thetaSq = theta.*theta;
%% Regularization term (remember to not include bias term)
JRegTerm = (lambda/(2*m))*sum(thetaSq(2:end));
%% Sum to get total cost
J = JNoReg + JRegTerm;

%% Calculate gradient with no regularization
gradNoReg = (1/m)*(X'*(hTheta - y));
%%gradNoReg = (1/m)*sum((hTheta-y)'*X)';
%% Copy theta and set first element to 0 (bias)
thetaReg = theta; thetaReg(1) = 0;
%% Then get regularization term
gradRegTerm = (lambda/m)*thetaReg;
%% Sum to get total grad
grad = gradNoReg + gradRegTerm;
		      
		     






% =========================================================================

grad = grad(:);

end
