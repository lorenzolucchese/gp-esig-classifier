import numpy as np
import math
import fbm


def BM_sample(begin, end, number, seed=None):
  np.random.seed(seed)
  timesteps = np.linspace(begin, end, number)
  s = (end-begin)/(number-1)
  sqrtS = math.sqrt(s)  
  sample = np.zeros(shape=(number))
  sample[0] = math.sqrt(begin) * np.random.randn(1)
  for i in range(1, number):
    sample[i] = sample[i-1] + sqrtS * np.random.randn(1)   
  
  return sample, timesteps


def FBM_sample(begin, end, number, H, seed=None):
  np.random.seed(seed)
  f = fbm.FBM(number - 1, H, length = end - begin)
  sample = f.fbm()
  timesteps = begin + f.times()

  return sample, timesteps


def GBM_sample(begin, end, InitCond, mu, sigma, number, seed=None): 
  np.random.seed(seed)
  timesteps = np.linspace(begin, end, number)
  s = (end - begin)/(number - 1)
  sqrtS = math.sqrt(s)
  const = mu - (sigma**2)/2
  sample = np.zeros(shape=(number))
  MB = np.zeros(shape=(number))
  sample[0] = InitCond
  MB[0] = math.sqrt(begin) * np.random.randn(1)
  for i in range(1, number):
    MB[i] = MB[i-1] + sqrtS * np.random.randn(1)
    sample[i] = InitCond * math.exp(const*(timesteps[i]) + sigma*(MB[i]))

  return sample, timesteps


def Sinh_sample(begin, end, InitCond, number, seed=None):
  np.random.seed(seed)
  timesteps = np.linspace(begin, end, number)
  s = (end - begin)/(number - 1)
  sqrtS = math.sqrt(s)
  sample = np.zeros(shape=(number))
  MB = np.zeros(shape=(number))
  sample[0] = InitCond
  constant = math.log(math.sqrt(1 + InitCond**2) + InitCond)
  MB[0] = math.sqrt(begin) * np.random.randn(1)
  for i in range(1, number):
    MB[i] = MB[i-1] + sqrtS*np.random.randn(1)
    sample[i] = math.sinh(constant + timesteps[i] + MB[i])

  return sample, timesteps


def OU_sample(begin, end, InitCond, alpha, gamma, beta, number, seed=None):
  np.random.seed(seed)
  timesteps = np.linspace(begin, end, number)
  s = (end - begin)/(number - 1)
  sqrtS = math.sqrt(s)
  sample = np.zeros(shape=(number))
  MB = np.zeros(shape=(number))
  sample[0] = InitCond
  MB[0] = math.sqrt(begin) * np.random.randn(1)
  StochIntApprox = 0
  for i in range(1, number):
    TimeVar1 = timesteps[i]
    MB[i] = MB[i-1] + sqrtS*np.random.randn(1)
    if i > 1:
       StochIntApprox += (math.exp(alpha * timesteps[i-1]) - math.exp(alpha * timesteps[i-2]))*(MB[i] - MB[i-1]) / (alpha*s) # This approximation comes from the theory of elementary processes
    sample[i] = InitCond * math.exp(-alpha*TimeVar1) + gamma * (1 - math.exp(-alpha*TimeVar1)) + beta * math.exp(-alpha*TimeVar1) * StochIntApprox

  return sample, timesteps