# Torch_Kron
Efficient 2D Kronecker product kernel

I made this in a couple of hours cause I needed a kernel that does this specific operation that I couldn't find online anywhere. GPT helped me a lil cause the Triton documentation is ok so don't expect this thing to be good.

The main purpose of this code is to take an arbitrary rank 4 tensor and do a self-kronecker on the last dimension s.t. the identity `(A • B)^n = K(A, n) ☉ K(B, n)` is true where K(X, n) is the nth order self-Kronecker product on X.
  
I am only doing 2-d in this code, so this reduces down to: `(A • B)^2 = K(A) ☉ K(B)` replacing K(X, 2) with K(X) and given A and B of d-dimensions.

If A and B or d-dimensional, then we can either do a sum of d terms, then square that, or do a sum of all pairwise terms. For example:
- `([a1 a2] • [b1 b2])^2 = (a1b1 + a2b2)^2 = (a1b1)^2 + a1b1a2b2 + a2b2a1b1 + (a2b2)^2 = (a1b1)^2 + 2*a2b2a1b1 + (a2b2)^2`
- `K([a1, a2]) ☉ K([b1 b2]) = [a1a1 a1a2 a2a1 a2a2] = [b1b1 b1b2 b2b1 b2b2] =(a1b1)^2 + a1b1a2b2 + a2b2a1b1 + (a2b2)^2 = (a1b1)^2 + 2*a2b2a1b1 + (a2b2)^2`
- However since we are doing an inner product, the output of the self-Kronecker K(...) can be reduced in dimensionality
- `K([a1, a2]) ☉ K([b1 b2]) = [a1a1 \sqrt(2)*a1a2 a2a2] = [b1b1 \sqrt(2)*b1b2 b2b2] = (a1b1)^2 + 2*a1b1a2b2 + (a2b2)^2`

The output dimension follows the multinomial theorem wrt. the dimension of the vector.

For example, a 3-d vector would result in a `(3^2=9)-d` self-Kronecker output vector. However due to the communative property of multiplication, we can reduce this vector down via the multinomial theorem to a 4-d output vector.


In general, for a d-dim vector, by the multinomial theorem, we can reduce the d^2-dim output vector to a `[(n+m-1) choose (m-1)]-dim` = `[(n+m-1)!/((m-1)!*n!)]-dim` vector where n is the power and m is the number of terms/dimension of the original vector the number of times we apply a Kronecker product. In this case, I am just doing 2 dimensions, so we are producing `[(d+1)!/(2*(d-1)!)]-dim` vectors which reduces further to simply `[d*(d+1)/2]-dim` vectors.
- so for 2d with a power of 2: `(2+2-1)!/((2-1)!*2!) = 3!/2! = 3 terms`
- and 3d with a power of 2: `(2+3-1)!/((3-1)!*2!) = 4!/(2!*2!) = 4*3*2/(2*2) = 6 terms`

For more Kronecker products (which is required for higher powers), the multinomial
  theorem should be used with a higher n
I'm using triton so this can be used much easier than straight cuda.