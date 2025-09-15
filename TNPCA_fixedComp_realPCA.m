function [V,U,D,XPrin,Xhat] = TNPCA_fixedComp_realPCA(X,R,Masking)
% X_tensor = tensor(X);
%[Vo,~,~,Uo] = hooi_popNet(X_tensor,R);
%clear X_tensor
% Masking is the network thresholding matrix

Xhat = X;
V = zeros(size(X,1),R); U = zeros(size(X,3),R); D = zeros(1,R);
XPrin = zeros(size(X));

% --- KEY CHANGE 1: Create a parallel.pool.Constant for the initial Xhat ---
Xhat_c = parallel.pool.Constant(Xhat);
for component = 1:R
    searchsize = 48;
    record_u = zeros(size(X,3),searchsize);
    record_v = zeros(size(X,1),searchsize);
    record_d = zeros(searchsize,1);
    VE = zeros(searchsize,1);
    if component == 1
        V_prev = [];
        U_prev = [];
    else
        V_prev = V(:,1:component-1);
        U_prev = U(:,1:component-1);
    end
    % --- KEY CHANGE 2: Inside the parfor, use the value from the constant ---
    parfor (search = 1:searchsize, 3)
        % Get the current data from the constant object
        CurrentXhat = Xhat_c.Value;
        v = GlorotInit(randn(size(CurrentXhat,1),1)); u = GlorotInit(randn(size(CurrentXhat,3),1));
        %v = Vo(:,component); u = Uo(:,component);
        v = v./norm(v); u = u./norm(u);
        M = Mult3D(CurrentXhat,applyProjection(v, V_prev),'node')
        obj = applyProjection(u, U_prev)'*M;
        objnew = obj;
        diff = 1;
        iter = 1;
        while diff > 10^(-6) && iter < 1000
            objold = objnew;
            Mu = Mult3D(CurrentXhat,u,'subject');
            eigfun_handler = @(x) applyProjection(Mu*applyProjection(x,V_prev),V_prev);
            [v,~] = eigs(eigfun_handler,size(CurrentXhat,1),1,'largestreal');
            Mv = Mult3D(CurrentXhat,v,'node');
            u = applyProjection(Mv, U_prev);
            u = u./norm(u);
            M_Pv = Mult3D(CurrentXhat,applyProjection(v, V_prev),'node');
            objnew = applyProjection(u, U_prev)'*M_Pv;
            diff = abs(objnew - objold)/obj;
            iter = iter + 1;
        end
        record_u(:,search) = u;
        record_v(:,search) = v;
        record_d(search) = v'*Mult3D(CurrentXhat,u,'subject')*v;
        VE(search) = u'*Mult3D(CurrentXhat,v,'node');
    end
    Location = find(VE == max(VE));
    d = record_d(Location(1));
    v = record_v(:,Location(1));
    u = record_u(:,Location(1));
    V(:,component) = v;
    U(:,component) = u;
    D(component) = d;
    V1 = v.*v';
    V1 = V1(:);
    P = reshape(V1*u',size(X)).*Masking;
    XPrin = XPrin + d*P;
    % --- KEY CHANGE 3: Update the main Xhat and the constant for the next component ---
    Xhat = Xhat - d*P;
    Xhat_c = parallel.pool.Constant(Xhat); % Update the constant with the new Xhat
    clear P record_u record_v record_d VE
end
end

function [M] = Mult3D(A,v,mode)
sz = size(A);
if strcmp(mode,'node') == 1
    intermediate = pagemtimes(A, v);
    M = squeeze(pagemtimes(v',intermediate)); % Result is [1, 37000], transpose to [37000, 1]
elseif strcmp(mode,'subject') == 1
    C = reshape(A,[],sz(3));
    M = reshape(C*v,sz(1:2));  
end
end

function [y] = applyProjection(x, basis)
%APPLYPROJECTION Applies the projection (I - basis*basis') to vector x.
%   y = (I - basis*basis') * x
%
%   Inputs:
%       x     : Input vector (or matrix) to be projected.
%       basis : Matrix whose columns form the basis to project out.
%               If basis is empty, the identity operation is performed (y = x).
%
%   Output:
%       y     : The projected vector.

if isempty(basis)
    y = x;
else
    % The mathematically efficient way: y = x - basis * (basis' * x)
    y = x - basis * (basis' * x);
end
end