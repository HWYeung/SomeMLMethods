function [Cluster,Transformation,F2,numberiterforhydra,ProbFunct ] = NCA_HYDRA(X,clustnum,CaseControl,regularisation,hydraiter,kmeansinit,lengthconsensus)
% Best to have CaseControl status sorted 
Cluster = zeros(size(X,1),1);

consensuscluster = zeros(sum(CaseControl == 1),lengthconsensus);
textprogressbar(['HYDRA running ' num2str(lengthconsensus) ' clustering: ']);
for consensus = 1:lengthconsensus
    textprogressbar(consensus*100/lengthconsensus);
    numberiterforhydra = 0;
    initclust = zeros(size(X,1),1);
    Caselength = sum(CaseControl);
    if kmeansinit == 1
        %[Lnew,F2,~,Gradnorm,~] = NCMML(X,CaseControl,128,regularisation,clustnum,20,20, 0.7,[],[],[]);
        %idx = kmeans(X(CaseControl == 1,:),clustnum,'Replicates',20);
        idx = KMeansInit(X(CaseControl == 1,:),clustnum);
    else
        idx = randsample(clustnum,Caselength,true);
    end
    initclust(CaseControl == 1) = idx;
    newclust = initclust;
    Lold = ones(clustnum,size(X,2));
    Lnew = zeros(clustnum,size(X,2));
    %getNMI = nmi(newclust,oldclust);
    while norm(Lnew-Lold,'fro')>1e-6 && numberiterforhydra < hydraiter
        numberiterforhydra = numberiterforhydra + 1;
        Lold = Lnew;
        oldclust = newclust;
        [Lnew,~,~,~,ProbFunct] = NCMML(X,oldclust,128,regularisation,clustnum,400,200, 0.7,[],Lold,exp([clustnum ones(1,clustnum)]));
        %idx = kmeans(X(CaseControl == 1,:)*L_best',clustnum,'Replicates',10);
        ProbCase = ProbFunct(X(CaseControl,:));
        newclust(CaseControl == 1) = (ProbCase(:,2:3) == max(ProbCase(:,2:3),[],2))*[1;2];
    end
    consensuscluster(:,consensus) = newclust(CaseControl == 1);
end
textprogressbar('done with generating clustering results');
cleanupObj = onCleanup(@() clear('textprogressbar')); 
disp('Running Consensus Clustering ... ');
ClusterCase=consensus_clustering(consensuscluster,clustnum);
Cluster(CaseControl ==1) = ClusterCase;
disp('Recalculating the Mahalanobis Metric ... ');
[L_best,F2,~,~,ProbFunct] = NCMML(X,Cluster,128,regularisation,clustnum,20000,1000, 0.7,[],[],exp([clustnum ones(1,clustnum)]));
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
