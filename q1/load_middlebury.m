% get dataset files
unzip("https://vision.middlebury.edu/flow/code/flow-code-matlab.zip")
unzip("https://vision.middlebury.edu/flow/data/comp/zip/other-gray-twoframes.zip")
unzip("https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip")

% Add functions to path
addpath('flow-code-matlab')

% extract ground truth
gt=readFlowFile('other-gt-flow/Grove2/flow10.flo');
whos gt 

% export ground truth
save('grove2_flo10', 'gt')

% These MAT files can be read in python using :
% import scipy.io
% mat = scipy.io.loadmat('file.mat')