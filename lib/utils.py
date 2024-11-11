import numpy as np
from scipy import optimize
import torch
import signatory

def phi(x, C, a):
  '''
  x: signature norm squared
  C: phi function parameter, it controls how strict the normalization is
  a: phi function parameter, tipically fixed to 1 in order to avoid too much hyperparameter
  '''
  if x > C:
    return C + (C**(1+a))*(C**(-a) - x**(-a))/a
  else:
    return x


def dilatation(x, C, a, M, d):
  '''
  x: an array with dimension batch x samples x signature
  C, a : phi function parameters
  M : truncation level (called L in the paper)
  d : dimension of the time series we are computing the signature of (!! if time augmentation is deployed, then the dimension is increased by 1)
  '''
  xNumpy = x.detach().cpu().numpy().astype(np.float64) 

  coefficients = np.zeros((xNumpy.shape[0], xNumpy.shape[1], (M+1)))
  normalizz = np.zeros((xNumpy.shape[0], xNumpy.shape[1], 1))

  for i in range(0, xNumpy.shape[0]):
      for j in range(0, xNumpy.shape[1]):
          normQuad = 1 + np.sum(xNumpy[i, j]**2) # signature norm squared
          coefficients[i, j, 0] = 1 - phi(normQuad, C, a)
          for k in range(1, (M+1)):
            coefficients[i, j, k] = np.sum(xNumpy[i, j, int(((d**k-1)/(d-1)-1)):int(((d**(k+1)-1)/(d-1)-1))]**2)
          def polin(input):
            xMonomials = np.zeros((M+1)) + 1
            for k in range(1, (M+1)):
                xMonomials[k] = input**(2*k)
            return np.dot(xMonomials, coefficients[i, j])
          normalizz[i, j] = optimize.brentq(polin, 0, 2) if all(np.isfinite(coefficients[i, j])) else 0.
          if normalizz[i, j] > 1:
            normalizz[i, j] = 1
  return torch.from_numpy(normalizz)


''' grad computation of dilatation function (corollary B.14 in the paper) '''
class Normalization(torch.autograd.Function):
    @staticmethod
    def forward(self, input, C, M, d, exponents, a): #a=1, input should have dimension batch x samples x length_sign, M is the truncation level, d the dimension of the timeseries, C normalization constant, exponents is useful to define correctly the dilatation
      '''
      input: an array batch x signature, it is x in the previous function
      C, M, d: as in the previous function
      exponents: a vector as long as the signature, it has d times 1, d^2 times 2,..., d^M times the value M
      '''
      norm = dilatation(input, C, a, M, d).to(input.device)
      self.save_for_backward(input, exponents)
      self.C = C
      self.M = M
      self.d = d
      self.norm = norm # batch x sample x 1
      return norm.to(torch.float32)

    @staticmethod
    def backward(self, grad_output):
      '''
      grad_output: grad of upper layers
      '''
      input, exponents = self.saved_tensors
      normToSquarePower = (self.norm**(2*exponents)).to(torch.float64) #batch x samples x length_sign
      normToSquarePowerMinus1 = (self.norm**(2*exponents-1)).to(torch.float64) # batch x length_sign
      denominator = torch.sum((input**2)*(normToSquarePowerMinus1), -1, keepdim=True).to(torch.float64) #batch x samples x 1
      inputnorm = ((1 + torch.sum(input**2, -1, keepdim=True))**(1/2)).to(torch.float64) # batch x samples x 1 , norm of the pre-normalized signature

      # computing phi derivative based on the 2 branches of the phi function
      phiDerivative = torch.zeros(input.shape[0], input.shape[1], 1, dtype=torch.float64).to(input.device) #batch x samples x 1
      
      # second branch
      index1 = torch.where(inputnorm[:, :, 0]> (self.C)**(1/2))[0]
      if len(index1) > 0:
        phiDerivative[index1, :] = 2*(self.C)**2/(inputnorm[index1,:]**3)
      
      # first branch
      index2 = torch.where(inputnorm[:, :, 0]<=(self.C)**(1/2))[0]

      if len(index2) > 0:
        phiDerivative[index2, :] = 2*inputnorm[index2,:]
      
      # Numerator
      Numerator = normToSquarePower - (phiDerivative/(2*inputnorm))*torch.ones(input.shape[0], input.shape[1], int((self.d**(self.M+1)-1)/(self.d-1)-1)).to(input.device) # batch x samples x length_sign
      Numerator = input*Numerator

      gradient = Numerator/denominator # batch x samples x length_sign
      return grad_output*gradient, None, None, None, None, None


''' Triangular: transformed a vector into a lower triangular matrix '''
class Triangular(torch.nn.Module):
  def __init__(self, dim, L2, x_indices, y_indices):
    '''
    dim: dimension of the starting time series
    L2: number of new time instants
    x_indices, y_indices: explained in the model construction (below)
    '''
    super(Triangular, self).__init__()
    self.dim = dim
    self.L2 = L2
    self.x_indices = x_indices
    self.y_indices = y_indices


  def forward(self, x): # x is of size batch x int((int(L2*(L2+1)/2)-int((L2-alp)*(L2-alp+1)/2))*(int(dim*(dim+1)/2)))
    A = torch.zeros((x.shape[0], self.L2*self.dim, self.L2*self.dim)).to(x.device)
    A[:, torch.Tensor.long(self.x_indices), torch.Tensor.long(self.y_indices)] = x
    return A


''' Preparation with time augmentation: combines original time series and values sampled, then applies time augmentation '''
class PreparationWithTimeAugmentation(torch.nn.Module):
  def __init__(self, order, timesteps_cut, dim, extended_order):
    '''
    dim: as in the previous function
    timesteps_cut: number of time steps, sum of known and new time steps
    order, extended_order: explained in the model
    '''
    super(PreparationWithTimeAugmentation, self).__init__()
    self.order = order
    self.extended_order = extended_order
    self.cut = timesteps_cut
    self.d = dim

  def forward(self, x, y):
    '''    
    x: starting time series
    y: new values sampled
    '''
    if y.ndim == 2:
      K = 1
      y = y.unsqueeze(-1)
    else:
      K = y.shape[2]

    timesteps = x[:, :self.cut] # time instants: before known and then new ones
    values = x[:, self.cut:, None] # starting time series
    values = torch.cat((values.expand(-1, -1, K), y), 1) # concatenate values with the new values
    
    # reorder values
    values = values[:, self.extended_order.type(torch.LongTensor)]
    values = values.reshape((values.shape[0], self.cut, self.d, K))
    
    # adding time component
    timesteps = timesteps[:, self.order]
    path = torch.cat((values, timesteps[:, :, None, None].expand(-1, -1, -1, K)), 2).squeeze(-1)
    return path
 

''' Compute expected signature, using correction term on martingale indices '''
class ExpectedSignature(torch.nn.Module):
    def __init__(self, M, d, C = None, a = 1, martingale_indices = None):
        super(ExpectedSignature, self).__init__()
        self.M = M
        self.d = d
        self.C = C
        self.a = a
        if self.C:
          # build the exponents useful for normalization procedure
          exponents = torch.ones(int((d**(M + 1) - 1)/(d - 1) - 1))
          seen = d
          for j in range(2, M + 1):
              exponents[seen:seen + d**j] = torch.ones(d**j) * j
              seen += d**j
          self.register_buffer('exponents', exponents)
        self.martingale_indices = martingale_indices
        self.sig_correction_indices = self.get_signature_correction_indices()

    def forward(self, path):    
        length = path.shape[-2]                                                                                                               # shape: (..., samples, length, d)
        signatures_stream = signatory.signature(path.reshape(-1, length, self.d), self.M, stream=True)                                        # shape: (..., samples, length - 1, d + ... + d**M)
        signatures_stream = signatures_stream.clone().reshape((*path.shape[:-2], *signatures_stream.shape[-2:]))                                      # shape: (..., samples, length - 1, d + ... + d**M)
        # pre-compute signature indices for efficiency
        signatures = signatures_stream[..., -1, :]                                                                                            # shape: (..., samples, d + ... + d**M)
        if self.martingale_indices:
            signatures_lower = signatures_stream[..., :-1, :-self.d**self.M]                                                                  # shape: (..., samples, length - 2, d + ... + d**(M-1))
            # pre-pend signature starting values at zero
            signatures_start = torch.cat([torch.zeros(*path.shape[:-2], 1, self.d**i).to(path.device) for i in range(1, self.M)], dim=-1)     # shape: (..., samples, 1, d + ... + d**(M-1))	
            signatures_lower = torch.cat([signatures_start, signatures_lower], dim=-2)                                                        # shape: (..., samples, length - 1, d + ... + d**(M-1))        
            
            # append 0-th order signature
            signatures_lower = torch.cat([torch.ones(*path.shape[:-2], length - 1, 1).to(path.device), signatures_lower], dim=-1)             # shape: (..., samples, length - 1, 1 + d + ... + d**(M-1))
            corrections = torch.einsum('...jk,...jl->...kl', signatures_lower, path[..., 1:, :] - path[..., :-1, :]).flatten(start_dim=-2)    # shape: (..., samples, d + ... + d**M)

            c_hat = (signatures.clone() * corrections).mean(axis=-2) / (corrections**2).mean(axis=-2)                                         # shape: (..., d + ... + d**M)
            signatures[..., self.sig_correction_indices] -= (c_hat.unsqueeze(-2) * corrections)[..., self.sig_correction_indices]             # shape: (..., samples, d + ... + d**M)
        else:
            pass
        if self.C:
          norm = Normalization.apply(signatures, self.C, self.M, self.d, self.exponents, self.a)
          signatures = signatures * (norm ** self.exponents[None, None, :])
        return signatures.mean(axis=-2)                                                                                                       # shape: (..., d + ... + d**M)

    def get_signature_correction_indices(self):
        if self.martingale_indices:
          if not isinstance(self.martingale_indices, list) or not all(isinstance(i, int) for i in self.martingale_indices) or not all(0 <= i < self.d for i in self.martingale_indices):
              raise ValueError(f'self.martingale_indices argument must be a list of integers between 0 and self.d={self.d}.')
          sig_indices = []
          start_index = 0
          for i in range(1, self.M + 1):
              # in each signature level, of size d**i, we want to extract j*d**(i-1):(j+1)*d**(i-1) for j in self.martingale_indices
              for j in self.martingale_indices:
                  sig_indices.extend(list(range(start_index+j*self.d**(i-1), start_index+(j+1)*self.d**(i-1))))
              start_index += self.d**i
          return sig_indices
        else:
           return []
        

# Evaluation function to calculate accuracy
def evaluate_accuracy(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradient calculation during evaluation
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Total samples
    
    accuracy = correct / total * 100  # Calculate accuracy as a percentage
    return accuracy


def to_tensor(X):
   if isinstance(X, np.ndarray):
      return torch.from_numpy(X)
   elif isinstance(X, torch.Tensor):
      return X


