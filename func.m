function [z, f] = func(x)
global N_elem_in_layer N_layers w w0 wn; % global variables
f = zeros(N_elem_in_layer,N_layers); % allocate memory to the outputs of the neural network
f(:,1) = tanh(x*w0)'; % first output of the neural network (first hidden layer)
for j=2:N_layers % formation of all other inputs of the neural network
    f(:,j) = tanh(w(:,:,j-1)'*f(:,j-1));
end
z = tanh(f(:,end)'*wn); % output of the last neuron of the neural network