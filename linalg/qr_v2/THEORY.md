# QR

A = QR, where Q is orthogonal (Q^TQ=I), and R is upper triangular.
- A doesn't need to be a square matrix. A[m,n] = Q[m,k] x R[k,n], where k=min(m,n) (i.e. also known as reduced QR). Since rank(A)<=min(m,n), we only need at most k=min(m,n) basis columns in Q to span A.

## Basic Householder reflection

Idea: gradually convert A into R by making each column of A becoming xxx and zeros
- 1st column: a single value then zeros
- 2nd column: 2 values, then zeros

How do we do this? For each step, we use **Householder reflection**.
- H = I - 2 uu^T, where u is unit vector (uu^T is outer product)
- or equivalently, H = I - tau vv^T, then v doesn't need to be normalized
- Hx **reflects** x in u direction. More specifically, u is the normal vector of the reflection plane.
- Why? Hx = x - 2 u(u^Tx). u^Tx is dot product, projection length of x onto u (u is unit vector). Hence, u(u^Tx) is the projection vector of x onto u. Minus 2 of that -> reflect x.

Householder properties
- Orthogonal: H^TH = HH = I - 4uu^T + 4(uu^T)(uu^T) = I
- Reflection preserves length: ||Hx|| = ||x||

For each QR decomposition step, given a column vector x, the problem is to choose Householder u such that Hx eliminates all entries of x except the first one.

x  = [x0, x1, x2, ...]^T
Hx = [ a,  0,  0, ...]^T

Since this is a reflection, we know that a = +-||x|| -> we know exactly what is the final Hx. The rest is simple. Since we know x and Hx have the same length, and we are doing a reflection, u ~ Hx - x i.e. the vector going from tip of x to tip of Hx.
- We can also confirm it mathematically: Hx - x = -2uu^Tx = -2(u^Tx)u
- We have 2 choices of a, either positive or negative x norm. To improve numerical stability, we choose **opposite sign** of x0, because 1st entry of Hx - x becomes (a - x0), so by having a opposite sign of x0, we avoid (a - x0) becoming a small number.
- LAPACK convention: w = x - Hx, then store v = w / w0. In other words, the vector difference, normalized by its first entry -> 1st entry becomes 1. The normalization factor tau = 2 / ||v||^2.
- There's a trick to avoid recomputing norm of v. Notice that ||v||^2 = 1/(x0-a)^2 [(x0-a)^2 + x1^2 + ...] = 1/(x0-a)^2 [x1^2 + ...] + 1. Hence, we can compute sum of squares without the 1st elem.

We then use this Householder vector to update A.

HA = A - tau v(v^TA)

where v^TA is vector-matrix product, and v(v^TA) is an outer product.

We can repeat for subsequent columns. Note that for next columns, we don't need to consider all entries of the vector. Let's say for the 2nd column, we can pick the following

y  = [y0, y1, y2, ...]^T
Hy = [y0,  b,  0, ...]^T

Hence, b = +-||y[1:]||. Effectively, it's the same previous step on y[1:]. Then, its Householder vector has the form

v1 = [0, 1, xxx, ...]^T

Because (y0-y0) cancels out, and we normalize by the first entry like previously (LAPACK convention). Interestingly, due to this, when we update A matrix, we only need to consider the sub bottom right matrix.

H1 = I - tau1 v1v1^T

Since v1's 1st entry is zero, the outer product (v1v1^T) has its 1st row and 1st column filled with zeros. Hence, H1 has the structure

[1, 0, ..., 0]
[0, x, ..., x]
[      ...,  ]
[0, x, ..., x]

Hence, H1A does not change the 1st row of A. 1st column of A is also not changed because its first column has the structure [x, 0, ..., 0]^T.

**Degenerate case** When ||x|| = 0 (all entries are zeros), we should skip Householder reflection step, and move on to the next column.

## Blocked Householder QR

All steps we have done so far operate on **per-column** basis. Notice that HA can be decomposed into H times column vectors of A.

A0 = [a0 | b0 | c0 | d0], where a0/b0/c0/d0 are column vectors
H0 = Householder(a0)
A1 = [R0 | H0b0 | H0c0 | H0d0]

H1 = Householder(H0b0)
A2 = [R0 | R1 | H1H0c0 | H1H0d0 ]

Thus, Householder(H0b0) can be computed without computing H0c0 and H0d0. We also see a dependency chain: each column only affects the remaining columns. This observation leads to interesting consequences: We can delay the application of H0, then apply H1H0 together. This is the key idea behind blocked Householder.

Given A[m,n], we choose block size b, then split A into b columns, called **panel**, and (n-b) columns, the trailing matrix

A = [A_{panel} | A_{trail}]

We compute Householder on the panel as usual, which is now a skinny matrix, and obtains b Householder vectors and b taus. The math says that, instead of applying each reflector one-by-one, we can apply all at once, and they have the form of:

H_{b-1} H_{b-2} ... H_0 = I - VTV^T

where V is the stack of column Householder vectors, shape [m,b]. T is a matrix with shape [b,b]. Applying everything to the trailing matrix:

A_{trail} = VTV^TA_{trail}

Hence, we can do matmul in the sequence of: W=V^TA -> W=TW -> W=VW. Once this is done, we continue with the bottom right corner A[b:,b:].

Next question is, how do we compute T? There are 2 facts we know about T (proof is outside the scope of this note):
- T is upper triangular.
- Diagonal of T is taus.

First, we compute all pair-wise dot product of Householder vectors

S = V^TV - shape [b,b]

Then we compute T column-by-column. Assuming we have set the diagonal to taus. The general idea is that, at column i:
- y = T[:i,:i] @ S[:i,i] - shape [i]
- T[:i,i] = -tau[i] * y

The first step is taking the previous T and do mat-vec product a row (or column) of S. Each thread can handle 1 row of T i.e. 1 element in the current column of T. Thus, we only incur O(b) loop to compute the dot product, and b loop to obtain b columns.
