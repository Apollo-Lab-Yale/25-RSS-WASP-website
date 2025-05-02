#### 1. Motivation (Use optimize.gif in clips)



- Motivation: In optimization scenarios, a **sequence** of derivates guides to find optimums. What if we can make use of the **coherence** of the sequence to accelerate the derivate computation at the current input? 

- Goal: compute sequence of approximate derivatives of computable function as **quickly** and **accurately** as possible
- Inputs: $f: \mathbb{R}^n \to \mathbb{R}^m$, and sequence of inputs, $x_k, x_{k+1}, x_{k+2}, ...$
- Outputs: a sequence of approximate derivatives at given inputs, $\hat{\frac{\partial f}{\partial x}}|_{x_{k}}, \hat{\frac{\partial f}{\partial x}}|_{x_{k+1}},\hat{\frac{\partial f}{\partial x}}|_{x_{k+2}}...$
- What we achieved: Bt default, derivative computation for each input only requires updating the jacobian-vector product on one tangent direction regardless of the dimension of the function's inputs and outpus. 



#### 2. Explanation of Our Method (Use spaces.gif in clips)

- Think derivate as  a map from a matrix of tangents $\Delta X$ to a matrix of jacobian-vector product $\Delta F$
  - $\frac{\partial f}{\partial x}|_{x_{k}} \Delta X=\Delta F$
  - Each column of $\Delta X$ and $\Delta F$ is independent. $\frac{\partial f}{\partial x}|_{x_{k}} [\Delta x_1 | \Delta x_2|...|\Delta x_r]=[\Delta f_1 | \Delta f_2|...|\Delta f_r]$
- Given $\Delta X$ and $\Delta F$ matrices of sufficient rank and size, we can directly solve the derivative $(\frac{\partial f}{\partial x}|_{x_{k}})^T=(\Delta X^T)^\dagger\Delta F^T$
- Visualization of the solution: 
  - Dark blue lines represent the **affine spaces** created from the corresponding columns from $(\Delta X^T)^\dagger\Delta F^T$. 
  - The $Z$ matrices are null space basis matrices, and $Y$ matrices represent free variables that parameterize the affine spaces.
  - The derivative must reside in the intersection of these affine spaces as illustrated by the yellow dot.



#### 3. Step by step 

- Step 1: Suppose we are given a new input $x_1$, and our goal is to approximate a new derivative matrix  $\frac{\partial f}{\partial x}|_{x_{1}}$. The groud truth derivative is represented by the shifted yellow dot.

- Step 2:  Instead of updating all the affine spaces by computing the jacobian-vector product in each column, which incurs function calls many times and can be computationally expensive, we just update one jacobian-vector product and its associated affine space as illustrated by the purple line.

- Step 3: Update this space while leave the others intersecting at the previous derivative solution, The true derivative (yellow dot) must lie in the updated affine space (purple line).

- **Step 4**: Here comes our key insight: By leveraging the prior affine spaces that left unupdated and the updated one, we can formulate the derivative approximation problem as a constrained least-square optimization.

  - $\frac{\partial f}{\partial x}|_{x_{k}} \approx \text{argmin}_{D^T} || \Delta X^TD^T - \hat{\Delta F} ||_F^2 \ s.t. \ \Delta x_i^T D^T = \Delta f_i^T$ 
  - $\hat{\Delta F}$ corresponds to the prior affine spaces, while $\Delta f_i$ in the constraint corresponds to the updated affine space.

  In the visualization, solving this optimization problem means finding the point on the updated affine space (purple line) that minimizes the Euclidean distances to other unupdated affine spaces. The result $D$

  - on the updated affine space
  - best aligns the other prior affine spaces

  In our paper, we derive the closed form solution (green dot) of this optimization using its KKT system:

  ${D^*}^T=A^{-1}(I-s_i^{-1}\Delta x_i \Delta x_i^T A^{-1})2\Delta X \hat{\Delta F}^T + s_i^{-1}A^{-1}\Delta x_i \Delta f_i^T$ with $A=2\Delta X \Delta X^T$, $s_i=\Delta x_i^T A^{-1} \Delta x_i$

- Step 5: The solution is then used to update other affine spaces that all spaces intersect at $D$.

  - $\hat{\Delta F} = D^* \Delta X$

- Step 6-8: The process iterates, cycling through the web of affine spaces, interleaving updates the approximates of derivative and $\hat{\Delta F}$ for each new input in the sequence.  

- Step 9: Our approach is also equipped with a error detection and correction mechanism. If we detect the approximation (green dot) is too far away from the ground truth (yellow dot), it can spend additional iterations on the current input until certain accuracy is achieved. If all affine spaces are updated with groud truth jacobian-vector products, the solution will always match the ground truth. 



#### 3. Results

- Eval 1

  The evalution bechmarks our WASP method against four other popular or related implementations of computing derivatives. The condtions are:

  - Reverse-mode automatic differentiation with PyTorch backend (abbreviated as RAD-PyTorch)
  - Finite-differencing with NumPy backend (abbreviated as FD)
  - Simultaneous Perturbation Stochastic Approximation with NumPy backend (abbreviated as SPSA)
  - Web of Affine Spaces Optimization with orthonormal $\Delta X$ matrix and NumPy backend (abbreviated as WASP-O).
  - Web of Affine Spaces Optimization with random, non-orthonormal $Delta X$ matrix and NumPy backend (abbreviated as WASP-NO).

  Through evaluating these conditions on computing derivatives of nested sine-cosine functions that resemble forward kinematic functions in robotics, we show that our methods offer significant speed-up while at the same time maintains reasonable accuracy. 

  For readers interested in further insights, supplementary results are provided in the Appendix (§X-C) in the paper. Conditions in this supplemental section include JAX implementations compiled for both CPU and GPU, as well as Rust-based implementations, allowing modifications, optimizations, and compilations to strive for maximum possible performance for each condition.

  

- Eval 2

  The evaluation involves using a root-finding procedure to get a quadruped robot to match its feet and endeffector to specified poses. Our WASP conditions converge much more quickly than the alternative approaches. As the analysis in our paper would suggest, the WASP procedure with an orthonormal $\Delta X$ basis converges even faster and more stably. 