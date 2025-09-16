function [V,U,D,XPrin,Xhat] = TNPCA_vectorised(X,R,Masking)
% X_tensor = tensor(X);
%[Vo,~,~,Uo] = hooi_popNet(X_tensor,R);
%clear X_tensor
% Masking is the network thresholding matrix

Xhat = single(X);
V = zeros(size(X,1),R); U = zeros(size(X,3),R); D = zeros(1,R);
XPrin = zeros(size(X));

for component = 1:R
    searchsize = 24;
    record_d = zeros(searchsize,1);
    if component == 1
        V_prev = [];
        U_prev = [];
    else
        V_prev = V(:,1:component-1);
        U_prev = U(:,1:component-1);
    end
    v = GlorotInit_search(randn(size(Xhat,1),searchsize));
    u = GlorotInit_search(randn(size(Xhat,3),searchsize));
    active_mask = true(1, searchsize);
    v = v ./ vecnorm(v, 2, 1);
    u = u ./ vecnorm(u, 2, 1);
    Pv_v = applyProjection(v, V_prev);
    M = Mult3D(Xhat,Pv_v,'node');
    obj = sum(applyProjection(u, U_prev).*M);
    objnew = obj;
    diff = ones(1,searchsize);
    outeriter = 1;
    while any(active_mask) && outeriter < 1000
        objold = objnew;
        % Update v for all active searches using power iteration
        Mu = Mult3D(Xhat, u(:,active_mask), 'subject');
        v(:,active_mask) = batched_null_power_iteration(Mu, V_prev, 1000, 1e-6);
        Mv = Mult3D(Xhat,v(:,active_mask),'node');
        u(:,active_mask) = applyProjection(Mv, U_prev);
        u(:,active_mask) = u(:,active_mask)./ vecnorm(u(:,active_mask), 2, 1);
        Pv_v = applyProjection(v(:,active_mask), V_prev);
        M = Mult3D(Xhat,Pv_v,'node');
        objnew(active_mask) = sum(applyProjection(u(:,active_mask), U_prev).*M);
        diff(active_mask) = abs(objnew(active_mask) - objold(active_mask)) ./ obj(active_mask);
        active_mask = diff > 1e-6;
        outeriter = outeriter + 1;
    end
    record_u = u;
    record_v = v;
    Mu = Mult3D(Xhat,u,'subject');
    Mv = Mult3D(Xhat,v,'node');
    VE = sum(u.*Mv);
    for search = 1:searchsize
        tempt_v = v(:,search);
        record_d(search) = tempt_v'*Mu(:,:,search)*tempt_v;
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
    clear P record_u record_v record_d VE
end
end

function [M] = Mult3D(A,v,mode)
sz = size(A);
if strcmp(mode,'node') == 1
    intermediate = pagemtimes(A, v);
    M = squeeze(sum(intermediate .* v, 1));
    if size(M, 2) > 1
        M = M';
    end
elseif strcmp(mode,'subject') == 1
    C = reshape(A,[],sz(3));
    M_temp = C * v;  % size: [n*n, s]
    % Reshape result to [n, n, s]
    M = reshape(M_temp, sz(1), sz(2), size(v, 2));
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
if size(y, 1) == 1
    y = y';
end
end

function [BetaInit] = GlorotInit_search(data)
sz = size(data,1);
Z = 2*rand(sz,size(data,2)) - 1;
bound = sqrt(6 / (sz+1));
BetaInit = bound*Z;
end

function v = batched_null_power_iteration(Mu, basis, num_iters, tol)
    % Mu: [n2, n1, searchsize] - precomputed matrices
    % basis: [n1, r] - orthonormal basis for the space to project away from
    % Returns: v [n1, searchsize] - dominant eigenvectors in null space
    
    [n2, n1, searchsize] = size(Mu);
    
    % Initialize random vectors in null space
    v = randn(n1, searchsize);
    v = applyProjection(v, basis);
    v = v ./ vecnorm(v, 2, 1);
    
    for iter = 1:num_iters
        v_old = v;
        
        % Ensure we stay in null space
        v_proj = applyProjection(v, basis);
        
        % Multiply by Mu for each search
        Mv = zeros(n2, searchsize);
        for s = 1:searchsize
            Mv(:,s) = Mu(:,:,s) * v_proj(:,s);
        end
        
        % Project result back to null space
        v_new = applyProjection(Mv, basis);
        v_new = v_new ./ vecnorm(v_new, 2, 1);
        
        % Check convergence
        diff = vecnorm(v_new - v_old, 2, 1);
        if max(diff) < tol
            break;
        end
        
        v = v_new;
    end
end