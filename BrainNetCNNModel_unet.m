function [BrainNetCNNGraph] = BrainNetCNNModel_unet(LayerStruct)
  %{
  The Layer Parameters Struct should contains what's needed for building the model.
  Otherwise it will use the default setting
  typical Layer Parameter struct should look like (in python dictionary format,
  matlab struct will be similar):
  Dict = {"InputShape": nn, 
  "E2E": {"numLayers": n, "numfilter": f1, "leaky": a1, "dropprob": p1}, 
  "E2N": {"numfilter": f2, "leaky": a2, "dropprob": p2},
  "N2G": {"numfilter": f3, "leaky": a3, "dropprob": p3},
  "Dense": {"numLayers": n, "numfilter": f4, "leaky": a4, "dropprob": p4},
  "Output": {"numfilter": f5, "modeltype": "regression"}
  "reg" : {"type": ridge, "decay": 0.0000005}}
  %}

if ~isfield(LayerStruct,"InputShape")
    error("Input shape not specified")
end

% set default struct
E2Estruct = struct("numLayers", 1, "numfilter", 32, "leaky", 0.2, "dropprob", 0.2);
E2Nstruct = struct("numfilter", 64, "leaky", 0.2, "dropprob", 0.2);
N2Gstruct = struct("numfilter", 128, "leaky", 0.0001, "dropprob", 0.01);
Densestruct = struct("numLayers", 1, "numfilter", 32, "leaky", 0.0001, "dropprob", 0.01);
DefaultStruct = struct("InputShape", LayerStruct.InputShape, "E2E", E2Estruct, ...
    "E2N", E2Nstruct, "N2G", N2Gstruct, "Dense", Densestruct, "Output", LayerStruct.Output);

% Sanity Check for the Struct
outerfields = fieldnames(DefaultStruct);
ModelStruct = struct();
for i = 1:numel(outerfields)
    if ~isfield(LayerStruct, outerfields{i})
        ModelStruct.(outerfields{i}) = DefaultStruct.(outerfields{i});
        continue
    end
    ModelStruct.(outerfields{i}) = LayerStruct.(outerfields{i});
    if ~isstruct(DefaultStruct.(outerfields{i}))
        continue
    end
    innerfields = fieldnames(DefaultStruct.(outerfields{i}));
    for j = 1:numel(innerfields)
        if ~isfield(LayerStruct.(outerfields{i}), innerfields{j})
            ModelStruct.(outerfields{i}).(innerfields{j}) = DefaultStruct.(outerfields{i}).(innerfields{j});
        end
    end
end

% Start model building
lgraph = layerGraph();
M = ModelStruct.InputShape;
tempLayers = [imageInputLayer([M M 1],"Name","imageinput","Normalization","none")];
lgraph = addLayers(lgraph,tempLayers);

if isscalar(ModelStruct.Dense.numfilter)
    ModelStruct.Dense.numfilter = repmat(ModelStruct.Dense.numfilter, 1, ModelStruct.Dense.numLayers);
end

if length(ModelStruct.Dense.numfilter) ~= ModelStruct.Dense.numLayers
    error("Length of Dense filters not matching the number of Dense layers specified")
end

% Make unet output layer

final_left = [
    convolution2dLayer([1 M], 1, "Name", "final_left")
    resize2dLayer("OutputSize",[M M], "Name", "final_resize_left")];
final_right = [
    convolution2dLayer([M 1], 1, "Name", "final_right")
    resize2dLayer("OutputSize",[M M], "Name", "final_resize_right")];
final_out = [additionLayer(2, "Name", "final_add")
    regressionLayer('Name','unet_output')];

lgraph = addLayers(lgraph,final_left);
lgraph = addLayers(lgraph,final_right);
lgraph = addLayers(lgraph,final_out);
lgraph = connectLayers(lgraph,"final_resize_left","final_add/in1");
lgraph = connectLayers(lgraph,"final_resize_right","final_add/in2");

% Stacking E2E layers, as well as constructing the decoder network at the
% same time
for e2e_num = 1:ModelStruct.E2E.numLayers
    num_str = string(e2e_num);
    e2e_left = [
        convolution2dLayer([1 M],ModelStruct.E2E.numfilter,"Name","e2e_left_" + num_str)
        resize2dLayer("OutputSize",[M M], "Name", "resize_left_" + num_str)];
    e2e_right = [
        convolution2dLayer([M 1],ModelStruct.E2E.numfilter,"Name","e2e_right_" + num_str)
        resize2dLayer("OutputSize",[M M], "Name", "resize_right_" + num_str)];
    e2e_out = [
        additionLayer(2,"Name","e2e_addition_" + num_str)
        batchNormalizationLayer
        leakyReluLayer(ModelStruct.E2E.leaky,'Name',"e2e_leaky_" + num_str)
        dropoutLayer(ModelStruct.E2E.dropprob,"Name","e2e_out_" + num_str)];
    lgraph = addLayers(lgraph,e2e_left);
    lgraph = addLayers(lgraph,e2e_right);
    lgraph = addLayers(lgraph,e2e_out);

    decode_e2e_left = [
        convolution2dLayer([1 M],ModelStruct.E2E.numfilter,"Name","decode_e2e_left_" + num_str)
        resize2dLayer("OutputSize",[M M], "Name", "decode_resize_left_" + num_str)];
    decode_e2e_right = [
        convolution2dLayer([M 1],ModelStruct.E2E.numfilter,"Name","decode_e2e_right_" + num_str)
        resize2dLayer("OutputSize",[M M], "Name", "decode_resize_right_" + num_str)];
    decode_e2e_out = [
        additionLayer(2,"Name","decode_e2e_addition_" + num_str)
        batchNormalizationLayer
        leakyReluLayer(ModelStruct.E2E.leaky,'Name',"decode_e2e_leaky_" + num_str)
        dropoutLayer(ModelStruct.E2E.dropprob,"Name","decode_e2e_out_" + num_str)];
    lgraph = addLayers(lgraph,decode_e2e_left);
    lgraph = addLayers(lgraph,decode_e2e_right);
    lgraph = addLayers(lgraph,decode_e2e_out);

    if e2e_num == 1
        lgraph = connectLayers(lgraph,"imageinput","e2e_left_1");
        lgraph = connectLayers(lgraph,"imageinput","e2e_right_1");
        lgraph = connectLayers(lgraph,"decode_e2e_out_1","final_left");
        lgraph = connectLayers(lgraph,"decode_e2e_out_1","final_right");
    else
        lgraph = connectLayers(lgraph,"e2e_out_" + string(e2e_num - 1),...
                               "e2e_left_" + num_str);
        lgraph = connectLayers(lgraph,"e2e_out_" + string(e2e_num - 1),...
                               "e2e_right_" + num_str);
        lgraph = connectLayers(lgraph,"decode_e2e_out_" + num_str, ...
                                "decode_e2e_left_" + string(e2e_num - 1));
        lgraph = connectLayers(lgraph,"decode_e2e_out_" + num_str, ...
                                "decode_e2e_right_" + string(e2e_num - 1));
    end
    lgraph = connectLayers(lgraph,"resize_left_" + num_str,...
                           "e2e_addition_"+num_str+"/in1");
    lgraph = connectLayers(lgraph,"resize_right_" + num_str,...
                           "e2e_addition_"+num_str+"/in2");
    lgraph = connectLayers(lgraph,"decode_resize_left_" + num_str,...
                           "decode_e2e_addition_"+num_str+"/in1");
    lgraph = connectLayers(lgraph,"decode_resize_right_" + num_str,...
                           "decode_e2e_addition_"+num_str+"/in2");
end

% Stacking E2N and N2G
e2n_n2g = [convolution2dLayer([1 M],ModelStruct.E2N.numfilter,"Name","e2n")
    batchNormalizationLayer
    leakyReluLayer(ModelStruct.E2N.leaky,'Name',"e2n_leaky")
    dropoutLayer(ModelStruct.E2E.dropprob,"Name","e2n_out")
    fullyConnectedLayer(ModelStruct.N2G.numfilter,"Name","n2g")
    leakyReluLayer(ModelStruct.N2G.leaky,'Name',"n2g_leaky")
    dropoutLayer(ModelStruct.N2G.dropprob,"Name","n2g_out")
    ];
lgraph = addLayers(lgraph,e2n_n2g);
lgraph = connectLayers(lgraph,"e2e_out_" + ModelStruct.E2E.numLayers,"e2n");

% Now the decoder for G2N, then N2E
g2n = [
    fullyConnectedLayer(M * ModelStruct.E2N.numfilter, "Name", "g2n_fc")
    leakyReluLayer(ModelStruct.N2G.leaky, "Name", "g2n_leaky")
    dropoutLayer(ModelStruct.N2G.dropprob, "Name", "g2n_dropout")
];
n2e_left = [functionLayer(@(X) reshape(X, [M, 1, ModelStruct.E2N.numfilter, size(X,4)]), "Name", "n2e_left")
    resize2dLayer("OutputSize",[M M], "Name", "n2e_resize_left")];
n2e_right = [functionLayer(@(X) reshape(X, [1, M, ModelStruct.E2N.numfilter, size(X,4)]), "Name", "n2e_right")
    resize2dLayer("OutputSize",[M M], "Name", "n2e_resize_right")];
n2e_out = [
        additionLayer(2,"Name","n2e_addition")
        batchNormalizationLayer
        leakyReluLayer(ModelStruct.E2E.leaky,'Name',"n2e_leaky")
        dropoutLayer(ModelStruct.E2E.dropprob,"Name","n2e_out")];
lgraph = addLayers(lgraph,g2n);
lgraph = addLayers(lgraph,n2e_left);
lgraph = addLayers(lgraph,n2e_right);
lgraph = addLayers(lgraph,n2e_out);
lgraph = connectLayers(lgraph,"g2n_dropout","n2e_left");
lgraph = connectLayers(lgraph,"g2n_dropout","n2e_right");
lgraph = connectLayers(lgraph,"n2e_resize_left","n2e_addition/in1");
lgraph = connectLayers(lgraph,"n2e_resize_right","n2e_addition/in2");
lgraph = connectLayers(lgraph,"n2e_out","decode_e2e_left_" + ModelStruct.E2E.numLayers);
lgraph = connectLayers(lgraph,"n2e_out","decode_e2e_right_" + ModelStruct.E2E.numLayers);

for dense_num = 1:ModelStruct.Dense.numLayers
    numstr = string(dense_num);
    dense = [fullyConnectedLayer(ModelStruct.Dense.numfilter(dense_num),"Name","dense_"+numstr)
        leakyReluLayer(ModelStruct.Dense.leaky,'Name',"dense_leaky_"+numstr)
        dropoutLayer(ModelStruct.Dense.dropprob,"Name","dense_out_"+numstr)];
    lgraph = addLayers(lgraph, dense);

    % Now construct the decoder dense layer part
    if dense_num == 1
        decode_dense_numfilter = ModelStruct.N2G.numfilter;
    else
        decode_dense_numfilter = ModelStruct.Dense.numfilter(dense_num);
    end

    decode_dense = [fullyConnectedLayer(decode_dense_numfilter,"Name","decode_dense_"+numstr)
        leakyReluLayer(ModelStruct.Dense.leaky,'Name',"decode_dense_leaky_"+numstr)
        dropoutLayer(ModelStruct.Dense.dropprob,"Name","decode_dense_out_"+numstr)];
    lgraph = addLayers(lgraph, decode_dense);

    if dense_num == 1
       lgraph = connectLayers(lgraph,"n2g_out","dense_1");
       lgraph = connectLayers(lgraph,"decode_dense_out_1","g2n_fc");
    else
       lgraph = connectLayers(lgraph,"dense_out_"+string(dense_num-1),"dense_"+numstr);
       lgraph = connectLayers(lgraph,"decode_dense_out_"+numstr,"decode_dense_"+string(dense_num-1));
    end
end

lgraph = connectLayers(lgraph,"dense_out_" + ModelStruct.Dense.numLayers, ...
    "decode_dense_" + ModelStruct.Dense.numLayers);
if ModelStruct.Output.modeltype ~= "none"
    if ModelStruct.Output.modeltype == "classification"
        Outputlayers = [fullyConnectedLayer(ModelStruct.Output.numfilter,"Name","out")
            softmaxLayer("Name","softmax")
            classificationLayer('Name','output')];
    elseif ModelStruct.Output.modeltype == "regression"
        Outputlayers = [fullyConnectedLayer(ModelStruct.Output.numfilter,"Name","out")
            regressionLayer('Name','output')];
    end
    
    lgraph = addLayers(lgraph, Outputlayers);
    lgraph = connectLayers(lgraph,"dense_out_" + ModelStruct.Dense.numLayers, "out");
end

BrainNetCNNGraph = lgraph;

disp("This is a " + ModelStruct.Output.modeltype + " model. Make sure you choose the correct loss function when training the model.")
disp("To view the model specification and visualising the model arcitecture, use functions analyzeNetwork and plot.")
disp("Regularisation parameters and optimiser should be specified in the trainnet function");
if length(BrainNetCNNGraph.OutputNames) > 1
    disp("A multi-output LayerGraph is constructed. In this case, you will need to use dlnetwork and a custom loop to train the network.")
end