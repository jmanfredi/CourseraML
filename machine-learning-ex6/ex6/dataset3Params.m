function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%%Define temporary variables
Ctemp = 0;
sigmaTemp = 0;

values = [0.01,0.03,0.1,0.3,1,3,10,30]

%%Loop over all 64 possible combinations
for i=1:8
  Ctemp = values(i);
  for j=1:8
    printf('Training model number: %f\n',(i-1)*8+j)
    sigmaTemp = values(j);
    model = svmTrain(X,y,Ctemp, @(x1,x2) gaussianKernel(x1,x2,sigmaTemp));
    %%Now, predict on the cross validation set
    predict = svmPredict(model,Xval);
    %%Evaluate model error
    cvError = mean(double(predict ~= yval));
    printf('CV Error is %f for C of %f and sigma of %f\n',cvError,Ctemp,sigmaTemp);
    if i>1 || j>1
      if cvError < minError
	minError = cvError;
	C = Ctemp;
	sigma = sigmaTemp;
      end
    else
      minError = cvError;
      C = Ctemp;
      sigma = sigmaTemp;
    end

  end
end

printf('Best value of C: %f\n',C);
printf('Best value of sigma: %f\n',sigma);




% =========================================================================

end
