import numpy as np
from scipy import optimize
import torch

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
  x: an array with dimension batch x signature
  C, a : phi function parameters
  M : truncation level (called L in the paper)
  d : dimension of the time series we are computing the signature of (!! if time augmentation is deployed, then the dimension is increased by 1)
  '''
  xNumpy = x.detach().numpy()

  coefficients = np.zeros((xNumpy.shape[0], (M+1)))
  normalizz = np.zeros((xNumpy.shape[0], 1))

  for i in range(0,xNumpy.shape[0]):
      normQuad = 1 + np.sum(xNumpy[i]**2) #signature norm squared
      coefficients[i, 0] = 1-phi(normQuad,C,a)
      for j in range(1, (M+1)):
         coefficients[i, j] = np.sum(xNumpy[i, int(((d**j-1)/(d-1)-1)):int(((d**(j+1)-1)/(d-1)-1))]**2)
      def polin(input):
        xMonomials = np.zeros((M+1)) + 1
        for k in range(1, (M+1)):
            xMonomials[k] = input**(2*k)
        return np.dot(xMonomials, coefficients[i])
      normalizz[i] = optimize.brentq(polin, 0, 2)
      if normalizz[i] > 1:
        normalizz[i] = 1
  return torch.from_numpy(normalizz)


''' grad computation of dilatation function (corollary B.14 in the paper) '''
class Normalization(torch.autograd.Function):
    @staticmethod
    def forward(self, input, C, M, d, exponents): #a=1, input should have dimension batch x length_sign, M is the truncation level, d the dimension of the timeseries, C normalization constant, exponents is useful to define correctly the dilatation
      '''
      input: an array batch x signature, it is x in the previous function
      C, M ,d: as in the previous function
      exponents: a vector as long as the signature, it has d times 1, d^2 times 2,..., d^M times the value M
      '''
      norm = dilatation(input, C, 1, M, d)
      self.save_for_backward(input, exponents)
      self.C = C
      self.M = M
      self.d = d
      self.norm = norm # batch x 1
      return norm.to(torch.float32)

    @staticmethod
    def backward(self, grad_output):
      '''
      grad_output: grad of upper layers
      '''
      input,exponents = self.saved_tensors
      normToSquarePower = (self.norm**(2*exponents)).to(torch.float64) #batch x length_sign
      normToSquarePowerMinus1 = (self.norm**(2*exponents-1)).to(torch.float64) #batch x length_sign
      denominator = torch.sum((input**2)*(normToSquarePowerMinus1),1,keepdim=True) #batch x 1
      inputnorm = (1+torch.sum(input**2,1,keepdim=True))**(1/2) # batch x 1 ,norm of the pre-normalized signature

      # computing phi derivative based on the 2 branches of the phi function
      phiDerivative = torch.zeros(input.shape[0], 1, dtype=torch.float64) #batch x 1
      
      # second branch
      index1 = torch.where(inputnorm[:,0]>(self.C)**(1/2))[0]
      if len(index1) > 0:
        phiDerivative[index1,:] = 2*(self.C)**2/(inputnorm[index1,:]**3)
      
      # first branch
      index2=torch.where(inputnorm[:,0]<=(self.C)**(1/2))[0]
      if len(index2) > 0:
        phiDerivative[index2,:] = 2*inputnorm[index2,:]
      
      # Numerator
      Numerator = normToSquarePower - (phiDerivative/(2*inputnorm))*torch.ones(input.shape[0], int((self.d**(self.M+1)-1)/(self.d-1)-1)) # batch x length_sign
      Numerator = input*Numerator

      gradient = Numerator/denominator # batch x length_sign
      return grad_output*gradient, None, None, None, None


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


  def forward(self,x): # x is of size batch x int((int(L2*(L2+1)/2)-int((L2-alp)*(L2-alp+1)/2))*(int(dim*(dim+1)/2)))
    A = torch.zeros((x.shape[0], self.L2*self.dim, self.L2*self.dim))
    A[:,torch.Tensor.long(self.x_indices), torch.Tensor.long(self.y_indices)] = x
    return A


''' Preparation with time augmentation: combines original time series and values sampled, then applies time augmentation '''
class PreparationWithTimeAugmentation(torch.nn.Module):
  def __init__(self, order, timesteps_cut, dim, extended_order):
    '''
    dim: as in the previous function
    timesteps_cut: number of time steps, sum of known an new time steps
    order, extended_order: explained in the model
    '''
    super(PreparationWithTimeAugmentation,self).__init__()
    self.order = order
    self.extended_order = extended_order
    self.cut = timesteps_cut
    self.d = dim

  def forward(self, x, y):
    '''    
    x: starting time series
    y: new values sampled
    '''
    timesteps = x[:, :self.cut] # time instants: before known and then new ones
    values = x[:, self.cut:] # starting time series
    values = torch.cat((values, y), 1) # concatenate values with the new values
    
    # reorder values
    values = values[:, self.extended_order.type(torch.LongTensor)]
    values = values.reshape([values.shape[0], self.cut, self.d])
    
    #adding time component
    timesteps = timesteps[:, self.order]
    path = torch.cat((values, timesteps.unsqueeze(2)), 2)
    return path