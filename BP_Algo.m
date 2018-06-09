function BP_Algo(XX,YY)
global NLenOfData N_layers NFactor w w0 wn N_elem_in_layer;
NLenOfData = length(XX(:,1)); % length of data for learning
%% form data for learning
N = fix(NLenOfData / 100);
x = []; T = [];
i = 1;
while length(T) < 100
    x = [x; XX(i,:)];
    T = [T; YY(i)];
    i = i + 1;
end
% parameters for neural network
NLenOfData = length(T);
N_layers = 10;
NFactor = length(XX(1,:));
N_elem_in_layer = 50;
nu = 0.03; % learning coefficient
% generate init weight matrixs
w0 = rand(NFactor, N_elem_in_layer)/10; % for first hidden layer
w  = rand(N_elem_in_layer, N_elem_in_layer, N_layers-1)/10; % for 2 ... n-1 layers 
wn = rand(N_elem_in_layer, 1)/10; % for last hidden layer

%% variables for determining the weighting of coefficients
dw0  = zeros(NFactor, N_elem_in_layer); % for first hidden layer
dw   = zeros(N_elem_in_layer, N_elem_in_layer, N_layers-1); % for 2 ... n-1 layers 
dwn  = zeros(N_elem_in_layer, 1); %  for last hidden layer

del = zeros(N_elem_in_layer,N_layers); %  matrix of increment determination

y_ = [];
%% start of learning
for i=1:NLenOfData 
    [ok, o] = func(x(i,:)); % determine the outputs of the neural network on the i-th test
    deln = T(i)-ok; % determined the deviations at the exit
    del(:,end) = (1+o(:,end)).*(1-o(:,end))*deln.*wn; %  determine the deviation in the penultimate layer
    for j = N_layers-1:-1:1  % loop through the layers
        del(:,j) = (1+o(:,j+1)).*(1-o(:,j+1)).*(del(:,j+1)'*w(:,:,j))';
    end
    % determine the corrected value for the first layer
    for j = 1:N_elem_in_layer
        dw0(:,j) = nu*del(j,1)*x(i,:)';
    end
    for k=1:N_layers-1 % passage through the layers of the neural network
        for j = 1:N_elem_in_layer
            dw(:,j,k) = nu*del(j,k+1)*o(j,k); % we adjust the weight coefficients in hidden layers
        end
    end
    dwn = nu*deln*o(:,end); % weight adjustment on the last layer
    %% taking into account changes in weight coefficients
    w0 = w0 + dw0;
    w = w + dw;
    wn = wn + dwn;
end