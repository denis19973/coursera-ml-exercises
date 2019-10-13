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

sigma_c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
min_c = 0;
min_sigma = 0;
min_error = NaN;

for cur_c = sigma_c_values
  for cur_s = sigma_c_values
    model= svmTrain(X, y, cur_c, @(x1, x2) gaussianKernel(x1, x2, cur_s));
    predictions = svmPredict(model, Xval);
    curr_error = mean(double(predictions ~= yval));
    if isnan(min_error)
      min_error = curr_error;
      min_c = cur_c;
      min_sigma = cur_s;
    elseif curr_error < min_error
      min_error = curr_error;
      min_c = cur_c;
      min_sigma = cur_s;
    endif
  endfor
endfor

C = min_c;
sigma = min_sigma;
% =========================================================================

end
