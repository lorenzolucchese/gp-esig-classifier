import torch
from lib.utils import Triangular, PreparationWithTimeAugmentation, ExpectedSignature


class GPES(torch.nn.Module):
  def __init__(self, L1, L2, dim, order, extended_order, alpha, level, number_classes, C, a, K, martingale_indices = None):
    '''
    L1: number of known time instants, i.e. length of the time series
    L2: number of new time instants
    order: in Preparationwithtimeaugmentation we concatenate the starting values and the sampled one, order reorganize them (FOR EXAMPLE L1=100, L2=99, THEN ORDER=[0,100,1,101,2,102,..] IF THE NEW TIME INSTANTS ARE THE MIDDLE POINTS)
    extended_order: as order but takes into account the dimension of the time series (FOR EXAMPLE D=2, L1=100,L2=99 SO STARTING TIME SERIES HAS 200 VALUES AND NEW VALUES ARE 198, SO EXTENDED ORDER=[0,1,200,201,2,3,202,203,...])
    alpha: how many subdiagonals of the lower triangular matrix are non zero, alpha='full' 'means no zero in the lower part
    level: signature truncation level
    number_classes: number of labels in the classification problem
    C, a: phi function parameters
    K: numbers of augmented paths generated
    dim: dimension of the starting time series
    '''
    super(GPES, self).__init__()

    # set alpha to numerical value    
    alpha = L2 if alpha == 'full' else alpha

    self.K = K
    self.C = C
    self.a = a
    self.alpha = alpha
    self.L1 = L1
    self.L2 = L2
    self.dim = dim
    self.order = order
    self.extended_order = extended_order

    if martingale_indices is not None and (not isinstance(martingale_indices, list) or not all(isinstance(i, int) for i in martingale_indices) or not all(0 <= i < dim for i in martingale_indices)):
      raise ValueError(f'martingale_indices={martingale_indices} must be a list of integers between 0 and dim={dim}.')

    # compute how much elements in the lower triangular matrix
    MatrixEl = int((int(L2*(L2+1)/2) - int((L2-alpha)*(L2-alpha+1)/2)) * (int(dim*(dim+1)/2)))
    self.level = level
    self.number_classes = number_classes

    # number of components of the signature (remember we have time augmentation)
    outputSigDim = int(((dim+1)**(level+1)-1)/(dim)-1)

    #needed to reshape first layer output into a matrix (in triangular function)
    tril_indices = torch.tril_indices(row=(dim), col=(dim), offset=0)
    x_indices = torch.zeros(int(L2*(L2+1)/2) - int((L2-alpha)*(L2-alpha+1)/2), dtype=torch.int32)
    for i in range(alpha):
      x_indices[i*L2-int(i*(i-1)/2):(i+1)*L2-int((i+1)*i/2)] = torch.arange(i, L2, 1)

    y_indices=torch.zeros(int(L2*(L2+1)/2) - int((L2-alpha)*(L2-alpha+1)/2), dtype=torch.int32)
    for i in range(alpha):
      y_indices[i*L2-int(i*(i-1)/2):(i+1)*L2-int((i+1)*i/2)] = torch.arange(0, L2-i, 1)

    self.x_indicesFull = torch.zeros(int((int(L2*(L2+1)/2) - int((L2-alpha)*(L2-alpha+1)/2))*(int(dim*(dim+1)/2))), dtype=torch.int32)
    self.y_indicesFull = torch.zeros(int((int(L2*(L2+1)/2) - int((L2-alpha)*(L2-alpha+1)/2))*(int(dim*(dim+1)/2))), dtype=torch.int32)
    for j in range(x_indices.shape[0]):
      self.x_indicesFull[(j*int(dim*(dim+1)/2)):((j+1)*int(dim*(dim+1)/2))] = (x_indices[j]*dim) + tril_indices[0]
      self.y_indicesFull[(j*int(dim*(dim+1)/2)):((j+1)*int(dim*(dim+1)/2))] = (y_indices[j]*dim) + tril_indices[1]

    self.MeanLayer = torch.nn.Linear((L1*(dim+1)+L2), (L2*dim)) # (L1 + L2 + L1*d, L2*d)
    self.SqrtCovLayer = torch.nn.Linear((L1*(dim+1)+L2), MatrixEl)
    self.FinalLayer = torch.nn.Linear(outputSigDim, number_classes)
    self.GaussianSampler = torch.distributions.Normal(0, 1)
    self.InterleaveWithTimeAugmentation = PreparationWithTimeAugmentation(order, L1 + L2, dim, extended_order)
    self.ExpectedSignatureLayer = ExpectedSignature(level, dim + 1, C=C, a=a, martingale_indices=martingale_indices)
    self.LogSoftmax = torch.nn.LogSoftmax(1)

  def forward(self, x):
    mean = self.MeanLayer(x)
    sqrtCov = self.SqrtCovLayer(x)
    sqrtCovMatrix = Triangular(self.dim, self.L2, self.x_indicesFull, self.y_indicesFull)(sqrtCov)
    epsilon = self.GaussianSampler.sample(torch.Size([x.shape[0], self.L2 * self.dim, self.K])).to(x.device)
    newValues = torch.bmm(sqrtCovMatrix, epsilon) + mean.unsqueeze(2)
    path = self.InterleaveWithTimeAugmentation(x, newValues).permute(0, 3, 1, 2)    
    self.MeanSig = self.ExpectedSignatureLayer(path)
    output = self.FinalLayer(self.MeanSig)
    output = self.LogSoftmax(output)
    return output