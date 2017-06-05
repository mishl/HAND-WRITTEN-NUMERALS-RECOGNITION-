% Michelle Lee 1000392915
clear variables;
clc;
clf;

% Organize data
load 'MNIST.mat';  % substitute the appropriate folder on your computer
n_epochs = 5;  % # passes through the whole training set
n_ex_per_mb = 100;  % # examples in each minibatch
n_minibatches = floor(60000/n_ex_per_mb);  % # minibatches in the whole training set
n_ex_quiz = 1000;  % # examples in each quiz (see line 42)


% Initialize network & graphics
n_layers = 3;
n_neurons = [784; 1000; 10];
o_mb = ones(1, n_ex_per_mb);  % used for minibatching, below
Q = rand(n_neurons(n_layers), n_neurons(1)) - 0.5;
[W, DW, b, Db, a, dL_dv_T,B] = deal(cell(n_layers, 1));  % more-compact code than in Backpropagation.m
eta = zeros(n_layers, 1);
ex_quiz= ones(1, n_ex_quiz);

for layer = 2:n_layers
  m = 0.1/sqrt(n_neurons(layer - 1));
  W{layer} = m*(rand(n_neurons(layer), n_neurons(layer - 1)) - 0.5);
  DW{layer} = zeros(n_neurons(layer), n_neurons(layer - 1));
  b{layer} = 0.1*rand(n_neurons(layer), 1);
  Db{layer} = zeros(n_neurons(layer), 1);
  eta(layer) = 5/n_neurons(layer - 1);  % smaller eta than in Backpropagation.m
  B{layer}=m*(rand(n_neurons(layer), n_neurons(layer - 1)) - 0.5);
end
mo = 0.9;
gt_step = ceil(n_minibatches/1000);
DATA = zeros(2, 1 + floor(n_minibatches/gt_step));
n = 0;
W{n_layers} = m*(rand(n_neurons(n_layers), n_neurons(n_layers - 1)) - 0.1);

% Train
for epoch = 1:n_epochs
    
  % Shuffle training data
  shuffle = randperm(size(TRAIN_images, 1));
  TRAIN_images = TRAIN_images(shuffle, :);
  TRAIN_labels = TRAIN_labels(shuffle, :);
  TRAIN_answers = TRAIN_answers(shuffle);

  for mb = 1:n_minibatches
  
    % Set inputs and desired outputs for this minibatch
    X = TRAIN_images(((mb-1)*n_ex_per_mb)+1:(mb)*(n_ex_per_mb),:)'; 
    Y_star = TRAIN_labels(((mb-1)*n_ex_per_mb)+1:(mb)*(n_ex_per_mb),:)'; 

    % Compute network's output and error
      a{1} = X;  % minibatch of first-layer activity vectors
  for layer = 2:n_layers - 1
    a{layer} = max(0, W{layer}*a{layer -1} + b{layer}*o_mb);  % middle layer
  end
  a{n_layers} = max(0,tanh(W{n_layers}*a{n_layers - 1} + b{n_layers}*o_mb));  % last layer
  Y_err = a{n_layers} - Y_star;

    % Adjust parameters
 dL_dv_T{n_layers} = 2*Y_err.* sech(a{n_layers}).^2;
  DW{n_layers} = mo*DW{n_layers} - eta(n_layers) * dL_dv_T{n_layers} * a{n_layers - 1}';  % summed effects of all examples in minibatch 
  Db{n_layers} = -eta(n_layers) * sum(dL_dv_T{n_layers}, 2);  % summed effects of all examples in minibatch
  for layer = n_layers - 1:-1:2
    dL_dv_T{layer} = (B{layer + 1}' * dL_dv_T{layer + 1}) .* sign(a{layer});
    DW{layer} = (mo*DW{layer} - eta(layer) * dL_dv_T{layer} * a{layer - 1}');  % summed effects of all examples in minibatch
    Db{layer} = mo*Db{layer} - eta(layer) * sum(dL_dv_T{layer}, 2); % summed effects of all examples in minibatch 
  end
  for layer = 2:n_layers
    W{layer} = W{layer} + DW{layer};
    b{layer} = b{layer} + Db{layer};
  end  
  
  % Record variables for plotting
  if mod(mb, gt_step) == 0
    n = n + 1;
    DATA(:, n) = [mb; mean(sum(Y_err.*Y_err))];  % store mean squared norm of error vectors in this minibatch
  end
    
  end  % for mb
    
  % Quiz the net, i.e. test it on a subset of test examples
  select = ceil(10000*rand(n_ex_quiz, 1)); 
  answers = TEST_answers(select, :)';
  
  %  test & compute your network's output, a{n_layers}
  a{1}= TEST_images(select, :)';
  for layer = 2:n_layers - 1
    a{layer} = max(0, W{layer}*a{layer -1} + b{layer}*ex_quiz);  % same biases for all examples in the minibatch
  end
  a{n_layers} = tanh(W{n_layers}*a{n_layers - 1} + b{n_layers}*ex_quiz);  % same biases for all examples in the minibatch
  
  % Count how many numerals were correctly identified
  [~, I] = max(a{n_layers});
  correct = sum((I - 1) == answers);
  disp([epoch, correct]);
    
end  % for epoch
