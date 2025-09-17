function [Cluster, Transformation, F2, numberiterforhydra, ProbFunct] = NCA_HYDRA_v2(X, ...
    clustnum, CaseControl, CV_set, hydraiter, kmeansinit, lengthconsensus)
% NCA_HYDRA
%   Performs subtype discovery with a hybrid approach combining 
%   Nearest Class Mean Metric Learning (NCMML) and HYDRA clustering.
%
% Inputs:
%   - X             : n x d data matrix (samples x features)
%   - clustnum      : number of subtypes to discover (including baseline/control cluster)
%   - CaseControl   : n x 1 binary vector indicating case (1) or control (0)
%                     Assumed sorted so cases come first or separated clearly
%   - regularisation: regularization scalar for metric learning
%   - hydraiter     : max iterations for HYDRA convergence
%   - kmeansinit    : flag (1 or 0) whether to initialize clusters with KMeans on case data
%   - lengthconsensus: number of consensus clustering runs for stability
%
% Outputs:
%   - Cluster           : n x 1 vector with cluster labels (1 = control/baseline, 2..k = subtypes)
%   - Transformation    : learned transformation matrix (L) from NCMML
%   - F2                : final training score from NCMML
%   - numberiterforhydra: number of iterations HYDRA ran for
%   - ProbFunct         : function handle to compute cluster probabilities for new data
%
% Note:
%   - Cluster assignment for cases is based on the max subtype probability excluding cluster 1.
%   - Consensus clustering is used to stabilize cluster assignments.

Cluster = zeros(size(X,1),1);

% Matrix to store clustering results for consensus runs (only for cases)
consensuscluster = zeros(sum(CaseControl == 1), lengthconsensus);

for consensus = 1:lengthconsensus
    numberiterforhydra = 0;
    initclust = zeros(size(X,1),1);
    Caselength = sum(CaseControl);

    % Initialization of clusters for cases
    if kmeansinit == 1
        % Initialize with KMeans on case data
        idx = KMeansInit(X(CaseControl == 1,:), clustnum);
    else
        % Random initialization for cases
        idx = randsample(clustnum, Caselength, true);
    end
    initclust(CaseControl == 1) = idx;
    initclust(CaseControl == 0) = 0;
    newclust = initclust;

    Lold = ones(clustnum, size(X,2));
    Lnew = zeros(clustnum, size(X,2));

    % HYDRA iterative updates
    while norm(Lnew - Lold, 'fro') > 1e-6 && numberiterforhydra < hydraiter
        numberiterforhydra = numberiterforhydra + 1;
        Lold = Lnew;
        oldclust = newclust;

        % Learn metric with NCMML using current cluster assignments
        [Lnew, ~, ~, ~, ProbFunct] = NCMML_v2(X, oldclust, 128, [], [], clustnum, [], [], [], CV_set, Lold, exp([clustnum ones(1, clustnum)]));

        % Compute subtype probabilities for case samples
        ProbCase = ProbFunct(X(CaseControl,:));

        % For CASES only: assign to subtype with max probability (excluding cluster 0)
        [~, maxIdx] = max(ProbAll(CaseControl == 1, 2:end), [], 2); % Look at columns 2:end
        newclust(CaseControl == 1) = maxIdx; % This gives clusters 1, 2, 3, ..., clustnum
    end

    consensuscluster(:, consensus) = newclust(CaseControl == 1);
end

disp('Running Consensus Clustering ... ');
ClusterCase = consensus_clustering(consensuscluster, clustnum);
Cluster(CaseControl == 1) = ClusterCase;

disp('Recalculating the Mahalanobis Metric ... ');
[L_best, F2, ~, ~, ProbFunct] = NCMML_v2(X, Cluster, 128, [], [], clustnum, 20000, 1000, [], CV_set, [], exp([clustnum ones(1, clustnum)]));

Transformation = L_best;

disp('Done');

end


function IDXfinal=consensus_clustering(IDX,k)
%Function performs consensus clustering on a co-occurence matrix
[n,~]=size(IDX);
cooc=zeros(n);
for i=1:n-1
    for j=i+1:n
        cooc(i,j)=sum(IDX(i,:)==IDX(j,:));
    end
    %cooc(i,i)=sum(IDX(i,:)==IDX(i,:))/2;
end
cooc=cooc+cooc';
L=diag(sum(cooc,2))-cooc;

Ln=eye(n)-diag(sum(cooc,2).^(-1/2))*cooc*diag(sum(cooc,2).^(-1/2));
Ln(isnan(Ln))=0;
[V,~]=eig(Ln);
try
    IDXfinal=kmeans(V(:,1:k),k,'emptyaction','drop','replicates',20);
catch
    disp('Complex Eigenvectors Found...Using Non-Normalized Laplacian');
    [V,~]=eig(L);
    IDXfinal=kmeans(V(:,1:k),k,'emptyaction','drop','replicates',20);
end

end

function cluster_init = KMeansInit(X,k)
FeatureSize = size(X,2);
numconsensus = min(floor(2*sqrt(FeatureSize)),50);
% get at least 25% features to compute the clustering
onefourth = floor(0.25*FeatureSize);
restofthat = floor(0.75*FeatureSize);
init_concensus = zeros(size(X,1),numconsensus);
for randomfeatset = 1:numconsensus
    kfeatures = randsample(restofthat,1) + onefourth;
    kfeatsubset = randsample(FeatureSize,kfeatures);
    idx = kmeans(X(:,kfeatsubset),k,'Replicates',20);
    init_concensus(:,randomfeatset) = idx;
end
cluster_init=consensus_clustering(init_concensus,k);

end

