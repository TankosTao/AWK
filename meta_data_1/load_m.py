 
import scipy.io as scio
 
dataFile = 'fd_pairs.mat'
data = scio.loadmat(dataFile)
print(data)