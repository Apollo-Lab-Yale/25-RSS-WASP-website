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



#### 4. Code Snippets 

- Python

  ```python
  """
  Minimal demonstration of the WASP derivative-estimation algorithm using a
  NumPy backend.
  
  The routine can also be executed on alternative array libraries (e.g.,
  PyTorch, JAX) through TensorLy’s backend abstraction.
  
  For the full reference implementation, see
  https://github.com/Apollo-Lab-Yale/apollo-py/blob/main/apollo_toolbox_py/apollo_py/apollo_py_differentiation/apollo_py_differentiation_tensorly/derivative_method_tensorly.py
  """
  
  from typing import Callable
  import numpy as np
  
  class DerivativeMethodWASP:
      def __init__(self, n: int, m: int, orthonormal: bool = True, d_ell=0.3, d_theta=0.3):
          """
          Parameters
          ----------
          n : int
              Input dimensionality of the target function *f*.
          m : int
              Output dimensionality of *f*.
          orthonormal : bool, optional
              If ``True`` (recommended), the finite-difference perturbations Δx are kept
              orthonormal. Set to ``False`` only for benchmarking or ablation studies.
          d_ell : float, optional
              Length-norm threshold used in the WASP error test (default = 0.3).
          d_theta : float, optional
              Angular threshold (radians) used in the WASP error test (default = 0.3).
          """
          
          self.n = n
          self.m = m
          self.orthonormal = orthonormal
          self.cache = WASPCache(n, m, orthonormal)
          self.num_f_calls = 0
          self.d_theta = d_theta
          self.d_ell = d_ell
          self.fixed_i = None
  
      def clear_cache(self):
          self.cache.i = 0
          self.cache.delta_f_t = np.eye(self.n, self.m)
  
      def derivative_raw(self, f:Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
          """
          Compute the raw Jacobian of *f* at *x* via the WASP scheme.
  
          Parameters
          ----------
          f : Callable[[numpy.ndarray], numpy.ndarray]
              The vector-valued function whose derivative is sought.
          x : numpy.ndarray
              Evaluation point, shape ``(n,)`` or ``(n, 1)`` depending on the backend.
  
          Returns
          -------
          numpy.ndarray
              The Jacobian matrix ``df/dx`` at *x*, shape ``(m, n)``.
          """
  
          self.num_f_calls = 0
          f_k = f(x)
          self.num_f_calls += 1
          epsilon = 0.00000001
          cache = self.cache
  
          while True:
              if self.fixed_i is None:
                  i = self.cache.i
              else:
                  i = self.fixed_i
  
              delta_x_i = cache.delta_x[:, i]
              x_k_plus_delta_x_i = x + epsilon * delta_x_i
              f_k_plus_delta_x_i = f.call(x_k_plus_delta_x_i)
              self.num_f_calls += 1
  
              delta_f_i = (f_k_plus_delta_x_i - f_k) / epsilon
              delta_f_i_hat = cache.delta_f_t[i, :]
              return_result = close_enough(delta_f_i, delta_f_i_hat, self.d_theta, self.d_ell)
  
              # Update delta_f_t with the new delta_f_i
              cache.delta_f_t[i]= delta_f_i
  
              # Get optimization matrices
              c_1_mat = cache.c_1[i]
              c_2_mat = cache.c_2[i]
              delta_f_t = cache.delta_f_t
  
              # Reshape for matrix multiplication
              delta_f_i = np.reshape(delta_f_i, (-1, 1))
  
              # Calculate the closed-form solution
              d_t_star = c_1_mat @ delta_f_t + c_2_mat @ delta_f_i.T
              d_star = d_t_star.T
  
              # Update delta_f_t for next iteration
              tmp = d_star @ cache.delta_x
              cache.delta_f_t = tmp.T
  
              # Update i for next iteration
              new_i = i + 1
              if new_i >= len(x):
                  new_i = 0
              cache.i = new_i
  
              # Return if accurate enough or all directions checked
              if return_result or self.num_f_calls == len(x) + 1:
                  return d_star
  
  class WASPCache:
      def __init__(self, n: int, m: int, orthonormal_delta_x: bool = True):
          self.i = 0
          self.delta_f_t = np.eye(n, m)
          delta_x = get_tangent_matrix(n, orthonormal_delta_x)
          self.c_1 = []
          self.c_2 = []
  
          # Calculate A and its inverse
          a_mat = 2.0 * delta_x @ delta_x.T
          a_inv_mat = np.linalg.inv(a_mat)
          eye = np.eye(n, n)
  
          # Precompute c_1 and c_2 matrices for each dimension
          for i in range(n):
              delta_x_i = delta_x[:, i:i + 1]
              s_i = delta_x_i.T @ a_inv_mat @ delta_x_i
              s_i_inv = 1.0 / s_i
  
              # Term 1 of the closed-form solution
              c_1_mat = a_inv_mat @ (eye - s_i_inv * delta_x_i @ delta_x_i.T @ a_inv_mat) @ (2.0 * delta_x)
  
              # Term 2 of the closed-form solution
              c_2_mat = s_i_inv * a_inv_mat @ delta_x_i
  
              self.c_1.append(c_1_mat)
              self.c_2.append(c_2_mat)
  
          self.delta_x = delta_x
  
  def get_tangent_matrix(n: int, orthonormal: bool) -> np.ndarray:
      t = np.random.uniform(-1, 1, (n, n))
      if orthonormal:
          U, S, VT = np.linalg.svd(t, full_matrices=True)
          delta_x = U @ VT
          return delta_x
      else:
          return t
  
  def close_enough(a: np.ndarray, b: np.ndarray, d_theta: float, d_ell: float):
      a_n = np.linalg.norm(a)
      b_n = np.linalg.norm(b)
  
      if a_n == 0.0 or b_n == 0.0:
          return False
  
      tmp = np.abs((np.dot(a, b) / (a_n * b_n)) - 1.0)
      if tmp > d_theta:
          return False
  
      if not b_n == 0.0:
          tmp1 = np.abs((a_n / b_n) - 1.0)
      else:
          tmp1 = 10000000.0
      if not a_n == 0.0:
          tmp2 = np.abs((b_n / a_n) - 1.0)
      else:
          tmp2 = 10000000.0
  
      if min(tmp1, tmp2) > d_ell:
          return False
  
      return True
  
  ```

  