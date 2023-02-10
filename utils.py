import numpy as np
import torch

def mask_generator(p_m, x):
  """Generate mask vector.
  
  Args:
    - p_m: corruption probability
    - x: feature matrix
    
  Returns:
    - mask: binary mask matrix 
  """
  mask = np.random.binomial(1, p_m, x.shape)
  return mask


def pretext_generator(m, x):  
  """Generate corrupted samples.
  
  Args:
    m: mask matrix
    x: feature matrix
    
  Returns:
    m_new: final mask matrix after corruption
    x_tilde: corrupted feature matrix
  """
  
  # Parameters
  no, dim = x.shape  
  # Randomly (and column-wise) shuffle data
  x_bar = np.zeros([no, dim])
  for i in range(dim):
    idx = np.random.permutation(no)
    x_bar[:, i] = x[idx, i]
    
  # Corrupt samples
  x_tilde = x * (1-m) + x_bar * m  
  # Define new mask matrix
  m_new = 1 * (x != x_tilde)

  return m_new, x_tilde



if __name__ == "__main__":
    x = np.random.randn(3, 28*28)
    p_m = 0.2
    mask = mask_generator(p_m, x)
    print(mask.shape)
    print(mask[:10])
    m_new,x_tilde = pretext_generator(mask, x)
    print(x_tilde.shape)
