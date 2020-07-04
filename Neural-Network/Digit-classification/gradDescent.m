function [nn_params_opt Jarr] = gradDescent(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda,alpha,iters),
 
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));              
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
Theta1Opt = Theta1;
Theta2Opt = Theta2;
Jarr = zeros(size(iters),1);           
for i=1:iters,
  nn_params_temp = [Theta1Opt(:);Theta2Opt(:)];
  [J grad] = nnCostFunction(nn_params_temp,input_layer_size, hidden_layer_size,num_labels, X, y, lambda);
  grad1 = reshape(grad(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));        
  grad2 = reshape(grad((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
              
  Theta1Opt = Theta1Opt - (alpha)*grad1;
  Theta2Opt = Theta2Opt - (alpha)*grad2;  
  J
  Jarr(i) = J;
end  
nn_params_opt = [Theta1Opt(:);Theta2Opt(:)];
  
endfunction
