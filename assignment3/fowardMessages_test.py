# test if forward messages are working. 
import numpy as np

D = 3
L1 = 0.1 
L2 = 0.2

pairwise_cost = np.double(np.abs(np.tile(np.arange(0,D),(D,1)) - np.reshape(np.arange(0,D), (D,1))))
pairwise_cost[pairwise_cost==1] = L1
pairwise_cost[pairwise_cost>1] = L2

###########################################################

unary_cost = np.array([[0.2, 0.7, 0.4],[0.9, 0.5, 0.5],[0.9, 0.7, 0.2]])

dim = unary_cost.shape[1]
msg = np.zeros((dim,D))
#unary_cost = np.array([[0.2],[0.9],[0.9]])

# for each x,y in H,W get the unary cost of size D and the pairwise cost of size D x D

for i in range(dim-1):
    msg[:,i+1] = np.min(unary_cost[:,i] + msg[:,i] + pairwise_cost, axis=1) 

print(msg)

############################################
print('############################################')

rows = 5
columns = 3

temp = np.repeat(pairwise_cost[np.newaxis,:,:], rows, axis=0)
f_ = np.repeat(temp[:,np.newaxis, :,:], columns, axis=1)

cv_ = np.repeat(unary_cost[np.newaxis, :, :], rows, axis=0)


msg = np.zeros(cv_.shape)
# msg:  H x W x D
# f_ :  H x W x D x D
# cv :  H x W x D

for i in range(dim-1):
    msg[:,i+1,:] = np.min(cv_[:,i,:,np.newaxis] + \
        msg[:,i,:,np.newaxis] + f_[:,i,:,:], axis=1)

msg = np.transpose(msg, (0,2,1))
print(msg)

