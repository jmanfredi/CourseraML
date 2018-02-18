function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%%First, make a matrix in which each row corresponds to a data point, and the
%%columns correspond to centroids. There should be a 1 in each row for the
%%centroid corresponding to that particular data point. All other entries
%%are 0.
selectorMatrix = (idx == (1:K));
%%Now, we need to divide by the total number of points in each centroid, since
%%really what we want is an average.
avgSelectorMatrix = selectorMatrix./sum(selectorMatrix);
%%Lastly, multiply this matrix by X to calculate the new centroids.
%%centroids = (idx == (1:K))' * X;
centroids = avgSelectorMatrix' * X;






% =============================================================


end

