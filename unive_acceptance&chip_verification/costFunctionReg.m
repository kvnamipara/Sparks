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
h=sigmoid(X*theta);
t1=log(h);
t2=-y.*t1;
t3=log(1-h);
t4=(1-y).*t3;
t5=t2-t4;
t5=sum(t5);
J=(1/m)*t5;
J=J+((lambda/(2*m))*(sum(theta.*theta)-theta(1)*theta(1)));

h1=(h-y);
grad=h1'*X;
grad=grad/m;
grad=grad'+((lambda/m)*theta);
grad(1)=grad(1)-((lambda/m)*theta(1));


% =============================================================

end
