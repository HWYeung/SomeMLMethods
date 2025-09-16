# Tensor Network PCA (TNPCA)

This repository implements a **partially symmetric tensor rank decomposition** for 3-way tensors.  
Given data `X ∈ ℝ^{n × n × m}`, we extract `R` components of the form

$$
\widehat{X} \;=\; \sum_{r=1}^{R} d_r \; v^{(r)} \otimes v^{(r)} \otimes u^{(r)},
$$

where
- `v^{(r)} ∈ ℝ^n` is a node loading vector,
- `u^{(r)} ∈ ℝ^m` is a subject loading vector,
- `d_r ∈ ℝ` is a scalar weight,
- and `⊗` denotes the outer product.

This is a **CP decomposition** with enforced symmetry in the first two modes (`n × n`), analogous to PCA on matrices but extended to a partially symmetric tensor setting.

---

## Objective (single component)

For a candidate pair `(v, u)` the rank-1 reconstruction is

$$
T(v,u)_{ijk} = v_i v_j u_k.
$$

We seek to maximize the contraction with the residual tensor `Xhat`:

$$
\mathcal{J}(v,u) = \langle Xhat,\, T(v,u)\rangle
= \sum_{i,j,k} Xhat_{ijk}\, v_i v_j u_k.
$$

Equivalently, defining

$$
M(u) = \sum_{k=1}^{m} u_k \, Xhat(:,:,k) \in \mathbb{R}^{n \times n},
$$

we have

$$
\mathcal{J}(v,u) = v^\top M(u) v.
$$

---

## Alternating updates

Each component is extracted using **alternating optimization**:

1. **Update v (node factor):**  
   Given `u`, find the dominant eigenvector of `M(u)`  
   (restricted to the orthogonal complement of previously extracted node factors).

2. **Update u (subject factor):**  
   Given `v`, set

   $$
   u_k \propto v^\top Xhat(:,:,k) v, \quad k=1,\dots,m,
   $$

   then normalize and project away previously extracted subject factors.

Iterations alternate until convergence of the objective.

---

## Deflation and orthogonality

After extracting `(v^{(r)}, u^{(r)}, d_r)`, we subtract its contribution from the residual:

$$
Xhat \;\leftarrow\; Xhat - d_r \, v^{(r)} \otimes v^{(r)} \otimes u^{(r)}.
$$

Future components are constrained to be orthogonal to earlier ones (both in node and subject modes), yielding a greedy PCA-like deflation scheme.

---

## Relationship to other methods

- For `m = 1` (single subject), the method reduces to spectral decomposition of a symmetric matrix:  
  each component is `d v vᵀ` as in standard PCA.
- For general `m`, the model is a **symmetric-in-first-two-modes CP decomposition**.  
  It is **not** Tucker/HOOI; instead, it is a sum of partially symmetric rank-1 terms.
- The projection/deflation steps make the method more PCA-like (orthogonal components), at the cost of not being a pure unconstrained CP fit.

---

## Pseudocode

```text
Xhat = X
for r = 1..R
    initialize multiple (v,u) seeds
    for each seed:
        repeat until convergence:
            v ← dominant eigenvector of M(u), projected
            u ← normalize([v' * Xhat(:,:,k) * v]_k), projected
    select best seed
    d_r = v' * (Σ_k u_k Xhat(:,:,k)) * v
    Xhat ← Xhat - d_r * (v ⊗ v ⊗ u)
end
