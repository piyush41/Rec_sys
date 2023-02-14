import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import random
from scipy.sparse import csr_matrix
import time

def CUR_decomposition(Input_arr,k):

    m, n = Input_arr.shape

    Input_arr_squared = np.square(Input_arr)
    P_row = np.sum(Input_arr_squared, axis=1)
    P_col = np.sum(Input_arr_squared, axis=0)

    total_sum = np.sum(P_row)

    P_row = P_row / total_sum
    P_col = P_col / total_sum

    row_normalize = np.sqrt(k * P_row)
    col_normalize = np.sqrt(k * P_col)
    
    row_Dist = np.random.choice(m, k, p = P_row)
    col_Dist = np.random.choice(n, k, p = P_col)


    C = Input_arr[:, col_Dist]
    R = Input_arr[row_Dist, :]

    for i in range(k):
        C[:, i] = C[:, i] / col_normalize[col_Dist[i]]
        R[i, :] = R[i, :] / row_normalize[row_Dist[i]]

    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            W[i][j] = Input_arr[row_Dist[i]][col_Dist[j]]
    
    X, E, Yt = np.linalg.svd(W)
    for i in range(E.shape[0]):
        if E[i] > 0:
            E[i] = 1 / E[i]
    Y = np.transpose(Yt)
    E = np.diag(E)
    Xt = np.transpose(X)
    U = Y.dot(E.dot(Xt))

    return C, U, R

def rmse(pred,truth):
    err = np.sqrt(np.sum(np.square(truth - pred)) / truth.size)

    return err

   


def DOSVD(Input_X,N_LF,var):
    
    U, sigma, Vt = svds(Input_X, k = N_LF)
    numerator=(np.square(sigma).sum())
    var.append((numerator/denom)*100)
    sigma = np.diag(sigma)
   
    return U,sigma,Vt

# movies_df=pd.read_csv('/data1/home/piyushmishra/DA/Recommendation_system /ml-latest-small/movies.csv')
ratings_df=pd.read_csv('../data/ratings.csv')
ratings_df = ratings_df.apply(pd.to_numeric)
n_users=ratings_df['userId'].nunique()
n_movies=ratings_df['movieId'].nunique()
X_df = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
X = X_df.values
timeSVD=[]
RMSE_list=[]
var=[]
Storage_SVD=[]


Storage_CUR=[]
RMSE_CUR=[]
timeCUR=[]

U, sigma, Vt = np.linalg.svd(X)
denom=(np.square(sigma).sum())
K_List=[]
for i in range(1,n_users,5):
    start = time.time()
    K_List.append(i)
    U,sigma,Vt=DOSVD(X,i,var)
    
    X_pred = np.dot(np.dot(U, sigma), Vt) 
    curr_storage_SVD = U.size * U.itemsize +  sigma.size * sigma.itemsize + Vt.size * Vt.itemsize
    
    Storage_SVD.append(curr_storage_SVD)
    RMSE_list.append(rmse(X_pred,X))
    end = time.time()
    timeSVD.append((end-start) )


    error2 = np.inf
    storage2=np.inf
    start = time.time()
    for j in range(3):
                A2, B2, C2 = CUR_decomposition(X,i)

                # error2_ = compute_RMSE(A2, B2, C2)
                computed_ratings = A2.dot(B2.dot(C2))
                error2_ = rmse(computed_ratings,X)

                # Sparse Representation of C and R matrix
                A2 = csr_matrix(A2)
                C2 = csr_matrix(C2)
                # Computing the storage requirement for CUR
                storage2_ = A2.data.nbytes + A2.indptr.nbytes + A2.indices.nbytes

                storage2_ += B2.size * B2.itemsize

                storage2_ += C2.data.nbytes + C2.indptr.nbytes + C2.indices.nbytes

                if error2_ < error2:

                    error2 = error2_

                    storage2 = storage2_
    end = time.time()
    timeCUR.append((end-start))
    RMSE_CUR.append(error2)
    Storage_CUR.append(storage2)



# Plotting Uncomment the below lines to plot all the 
# plots given in the report.



# plt.plot(K_List,var)
# plt.xlabel("Number of Latent Factors")
# plt.ylabel("Variance Preserved in percentage")
# plt.grid()
# plt.legend()
# plt.show()

# plt.plot(K_List,RMSE_list)
# plt.xlabel("Number of Latent Factors")
# plt.ylabel("RMSE ERROR")

# plt.legend()
# plt.show()


# plt.plot(K_List[35:],RMSE_CUR[35:])
# plt.xlabel("Number of Latent Factors")
# plt.ylabel("RMSE ERROR")

# plt.legend()
# plt.show()


# plt.plot(K_List,np.divide(Storage_SVD,1000000),label='SVD')
# plt.plot(K_List,np.divide(Storage_CUR,1000000),label="CUR")
# plt.xlabel("Number of latent factors")
# plt.ylabel("Storage in MegaBytes")
# plt.legend()
# plt.show()


# plt.plot(K_List,timeSVD,label='SVD')
# plt.plot(K_List,timeCUR,label="CUR")
# plt.xlabel("Number of latent factors")
# plt.ylabel("Time in seconds")
# plt.legend()
# plt.show()






############################ Question 6 ################################################
ratings_df=pd.read_csv('../data/ratings.csv')
n_users=ratings_df['userId'].nunique()
n_movies=ratings_df['movieId'].nunique()
ratings_df = ratings_df.apply(pd.to_numeric)
rating=ratings_df.values
np.random.shuffle(rating)


def rmse1(pred,truth):
    row=pred.shape[0]
    col=pred.shape[1]
    error=0
    count=0
    for  i,j in zip(*truth.nonzero()):
        error+=((pred[i][j]-truth[i][j])**2)
        count=count+1
    error=error/(count)
    return np.sqrt(error)


def mf(R, k, n_epoch=100, lr=.0003, l2=.04):
  training_err=[]
  testing_err=[]
  
  tol = .001  
  
  m, n = R.shape
  P = np.random.rand(m, k)
  Q = np.random.rand(n, k)
  for epoch in range(n_epoch):
    count=0
    for u, i in zip(*R.nonzero()):
      count=count+1
      err_ui = R[u,i] - P[u,:].dot(Q[i,:])
      for j in range(k):
        P[u][j] += lr * (2 * err_ui * Q[i][j] - l2 * P[u][j])
        Q[i][j] += lr * (2 * err_ui * P[u][j] - l2 * Q[i][j])
    E = (R - P.dot(Q.T))**2
    obj = E[R.nonzero()].sum() + l2*((P**2).sum() +(Q**2).sum())
    pred=np.dot(P,Q.T)
    Rmse=rmse1(pred,R)
    testRmse=rmse1(pred,final_test)
    training_err.append(Rmse)
    testing_err.append(testRmse)
    if obj < tol:
        break

  # Uncomment the below lines to get the plot for PQ Decomposition

  # plt.plot(training_err,label="Training")
  # plt.plot(testing_err,label="Testing")
  # plt.xlabel("Epochs")
  # plt.ylabel("RMSE Error")
  # plt.legend()
  # plt.show()
  
  return P, Q

split_size=int(0.8*rating.shape[0])
X_train=rating[:split_size,:]
X_test=rating[split_size:,:]
X_train_df = pd.DataFrame(X_train, columns = ratings_df.columns)
X_test_df = pd.DataFrame(X_test, columns = ratings_df.columns)

final_train=np.zeros((n_users,n_movies))
final_test=np.zeros((n_users,n_movies))

moviedict={}
moviecount=0
for i in range(ratings_df.shape[0]):
    if ratings_df.iloc[i]['movieId'] not in moviedict.keys():
        moviedict[ratings_df.iloc[i]['movieId']]=moviecount
        moviecount=moviecount+1
for i in range(X_train_df.shape[0]):
    user=int(X_train_df.iloc[i]['userId'])
    movie=X_train_df.iloc[i]['movieId']
    final_train[user-1][moviedict[movie]]=X_train_df.iloc[i]['rating']
for i in range(X_test_df.shape[0]):
    user=int(X_test_df.iloc[i]['userId'])
    movie=X_test_df.iloc[i]['movieId']
    final_test[user-1][moviedict[movie]]=X_test_df.iloc[i]['rating']

K=50
P,Q=mf(final_train,K,n_epoch=100)








####################################### Question 7 ################################

# movies_df=pd.read_csv('/data1/home/piyushmishra/DA/Recommendation_system /ml-latest-small/movies.csv')
ratings_df=pd.read_csv('../data/ratings.csv')
n_users=ratings_df['userId'].nunique()
n_movies=ratings_df['movieId'].nunique()
ratings_df = ratings_df.apply(pd.to_numeric)
rating=ratings_df.values
np.random.shuffle(rating)

def rmse1(pred,truth):
    row=pred.shape[0]
    col=pred.shape[1]
    
    error=0
    count=0
    for  i,j in zip(*truth.nonzero()):
        error+=((pred[i][j]-truth[i][j])**2)
        count=count+1
    error=error/(count)
    return np.sqrt(error)


split_size=int(0.8*rating.shape[0])
X_train=rating[:split_size,:]
X_test=rating[split_size:,:]
X_train_df = pd.DataFrame(X_train, columns = ratings_df.columns)
X_test_df = pd.DataFrame(X_test, columns = ratings_df.columns)
n_training=X_train_df.shape[0]
n_testing= X_test_df.shape[0]


final_train=np.zeros((n_users,n_movies))
final_test=np.zeros((n_users,n_movies))
moviedict={}
moviecount=0
for i in range(ratings_df.shape[0]):
    if ratings_df.iloc[i]['movieId'] not in moviedict.keys():
        moviedict[ratings_df.iloc[i]['movieId']]=moviecount
        moviecount=moviecount+1
randomcount=0
for i in range(X_train_df.shape[0]):
    user=int(X_train_df.iloc[i]['userId'])
    movie=X_train_df.iloc[i]['movieId']
    final_train[user-1][moviedict[movie]]=1
while True:
    rowno=random.randint(0,n_users-1)
    colno=random.randint(0,n_movies-1)
    if final_train[rowno][colno]==0:
        final_train[rowno][colno]=-1
        randomcount=randomcount+1
    if randomcount>=n_training:
        break
randomcount=0
for i in range(X_test_df.shape[0]):
    user=int(X_test_df.iloc[i]['userId'])
    movie=X_test_df.iloc[i]['movieId']
    final_test[user-1][moviedict[movie]]=1
while True:
    rowno=random.randint(0,n_users-1)
    colno=random.randint(0,n_movies-1)
    if final_test[rowno][colno]==0:
        final_test[rowno][colno]=-1
        randomcount=randomcount+1
    if randomcount>=n_testing:
        break
class NCFModel1:
  
  def __init__(self,factor=50):
    
    # self.input=np.concatenate((u_arr,m_arr),axis=1)
    self.factor=factor 
    self.W1 = np.random.randn(610,factor)
    self.W2 = np.random.randn(9724,factor)
    self.W3 = np.random.randn(2*factor,factor)
    self.W4 = np.random.randn(factor,factor)
    self.W5= np.random.randn(factor,1)
    
  
  def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))
  


  
  def forward_pass(self, u_arr,m_arr):
    # x = x.reshape(1, -1) # (1, 2)
    self.A1 = np.matmul(u_arr,self.W1) #+ self.B1  # (1, 610) * (610, K) -> (1, K)
    self.H1 = self.sigmoid(self.A1) # (1, K)
    self.A2 = np.matmul(m_arr, self.W2) #+ self.B2 # (1, 9724) * (9724, K) -> (1, K) 
    self.H2 = self.sigmoid(self.A2) # (1, K)
    self.concat = np.concatenate((self.H1,self.H2),axis=1)       #(1,2*k)
    self.A3 = np.matmul(self.concat, self.W3) #+ self.B3   # (1, 2*k) * (2*k, K) -> (1, K) 
    self.H3 = self.sigmoid(self.A3)              # (1,K)
    self.A4 = np.matmul(self.H3, self.W4) #+ self.B3   # (1, 2*k) * (2*k, K) -> (1, K) 
    self.H4 = self.sigmoid(self.A4)
    self.A5 = np.matmul(self.H4, self.W5) #+ self.B4 
    self.H5 = self.sigmoid(self.A5)

    return self.H5


  def getsparse(self,index,size):
    arr=np.zeros(size)
    arr[index]=1
    arr=arr.reshape(1,-1)
    return arr
    
  def grad_sigmoid(self, x):
    return x*(1-x) 
  
  def grad(self, u_arr, m_arr,rating):
    self.forward_pass(u_arr,m_arr)
    
    
    self.dH5 = ((self.H5 - rating)/(self.H5*(1-self.H5))) # (1, 1) 
    self.dA5 = np.multiply(self.dH5, self.grad_sigmoid(self.H5))  #(1,1)
    self.dW5 = np.matmul(self.H4.T, self.dA5)   #(K,1)


    self.dH4 = np.matmul(self.dA5, self.W5.T)   #(1,k)
    self.dA4 = np.multiply(self.dH4, self.grad_sigmoid(self.H4))
    self.dW4 = np.matmul(self.H3.T, self.dA4)

    
    self.dH3 = np.matmul(self.dA4, self.W4.T)   #(1,k)
    self.dA3 = np.multiply(self.dH3, self.grad_sigmoid(self.H3))
    self.dW3 = np.matmul(self.concat.T, self.dA3)
    movie_mat=self.dW3[self.factor:,]
    user_mat=self.dW3[:self.factor,]


    self.dH2 = np.matmul(self.dA3, movie_mat.T)   #(1,k)
    self.dA2 = np.multiply(self.dH2, self.grad_sigmoid(self.H2))
    self.dW2 = np.matmul(m_arr.T, self.dA2)

    self.dH1 = np.matmul(self.dA3, user_mat.T)   #(1,k)
    self.dA1 = np.multiply(self.dH1, self.grad_sigmoid(self.H1))
    self.dW1 = np.matmul(u_arr.T, self.dA1)

  def loss(self,y_true,y_pred):
      first= y_true*np.log(y_pred)
      second=(1-y_true)*np.log(1-y_pred)
      return -(first+second)

  def BinaryCrossEntropy(self,y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

  def fit(self, R, epochs=20, learning_rate=0.04, display_loss=False):
    m=R.shape[0]
    n=R.shape[1]
    self.loss_epoch=[]
    
    for i in range(epochs):
      dW1 = np.zeros((m,self.factor))
      dW2 = np.zeros((n,self.factor))
      dW3 = np.zeros((2*self.factor,self.factor))
      dW4 = np.zeros((self.factor,self.factor))
      dW5=np.zeros((self.factor,1))
      
      count=0
      true=[]
      pred=[]
      # for u, i in zip(*R.nonzero()):
      for u in range(m):
        for i in range(n):
          if R[u][i]==1 or R[u][i]==-1:
            curr_rat=R[u][i]
            if(R[u][i]==-1):
              curr_rat=0
            
            u_arr=self.getsparse(u,m)
            m_arr=self.getsparse(i,n)
            # closs+=self.loss(R[u,i],self.forward_pass(u_arr,m_arr))
            true.append(curr_rat)
            
            self.grad(u_arr, m_arr,curr_rat)
            pred.append(self.H5.item())
            dW1 += self.dW1
            dW2 += self.dW2
            dW3 += self.dW3
            dW4 += self.dW4  
            dW5+=self.dW5
            count=count+1
      self.W5 -=learning_rate*(dW5/count)
      self.W2 -= learning_rate * (dW2/(count))
      self.W3 -= learning_rate * (dW3/(count))
      self.W1 -= learning_rate * (dW1/(count))
      self.W4 -= learning_rate * (dW4/(count))

      loss=self.BinaryCrossEntropy(np.array(true).reshape(-1, 1), 
                         np.array(pred).reshape(-1, 1))
      
      self.loss_epoch.append(loss)      



out1=NCFModel1(50)
out1.fit(final_train)



###### for plotting ##############

# plt.plot(out1.loss_epoch)
# plt.xlabel("Epochs")
# plt.ylabel("Binary Cross Entropy Loss")
# plt.show()



def infer(R):
    true=[]
    pred=[]
    m=R.shape[0]
    n=R.shape[1]
    for u in range(m):
        for i in range(n):
            if R[u][i]==1 or R[u][i]==-1:
                curr_rat=R[u][i]
                if(R[u][i]==-1):
                    curr_rat=0
            
                u_arr=out1.getsparse(u,m)
                m_arr=out1.getsparse(i,n)
                # closs+=self.loss(R[u,i],self.forward_pass(u_arr,m_arr))
                true.append(curr_rat)
                
                H5=out1.forward_pass(u_arr, m_arr)
                pred.append(H5.item())
    print(out1.BinaryCrossEntropy(np.array(true).reshape(-1, 1), 
                         np.array(pred).reshape(-1, 1)))


# infer(final_test)



    
