import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# generates the ratings matrix
# rows represent users and columns represent movies
# empty cells are rated as 0
def get_ratings_matrix():
    users = df['userId'].unique().tolist()
    movies = df['movieId'].unique().tolist()
    users_idx = {}
    movies_idx = {}
    for i, user in enumerate(users):
        users_idx[user] = i
    for i, movie in enumerate(movies):
        movies_idx[movie] = i
    
    m, n = len(users), len(movies)
    ratings = np.zeros((m, n))
    for i in range(df.shape[0]):
        user, movie, rating = df.iloc[i]['userId'], df.iloc[i]['movieId'], df.iloc[i]['rating']
        ratings[users_idx[user]][movies_idx[movie]] = rating
    
    return ratings, users_idx, movies_idx

# To plot various metrics
def plot_metrics(K, error_svd, space_svd, time_svd, error_cur, space_cur, time_cur):
    
    plt.plot(K, time_svd, label = 'SVD')
    plt.plot(K, time_cur, label = 'CUR')
    plt.xlabel('#Latent Factors')
    plt.ylabel('Time Taken(s)')
    plt.title("Compute Time Vs Latent Factors")
    plt.legend()
    plt.show()

    plt.plot(K, np.array(space_svd)/1024/1024, label = 'SVD')
    plt.plot(K, np.array(space_cur)/1024/1024, label = 'CUR')
    plt.xlabel('#Latent Factors')
    plt.ylabel('Storage(MB)')
    plt.title("Storage Vs Latent Factors")
    plt.legend()
    plt.show()

    plt.plot(K, error_svd)
    plt.xlabel('#Latent Factors')
    plt.ylabel('Root Mean Squared Error')
    plt.title("Singular Valued Decomposition")
    plt.show()

    plt.plot(K[17:], error_cur[17:])
    plt.xlabel('#Latent Factors')
    plt.ylabel('Root Mean Squared Error')
    plt.title("CUR Decomposition")
    plt.show()


# Computes RMSE for SVD and CUR Decomposition
def compute_RMSE(A, B, C):
    computed_ratings = A.dot(B.dot(C))
    err = np.sqrt(np.sum(np.square(ratings - computed_ratings)) / ratings.size)
    return err

def SVD(mat, k, option = 1):
    if option == 1:
        # SVD with min(mat.shape) latent factors
        return np.linalg.svd(mat)
    else: 
        # SVD with k latent factors
        return svds(mat, k = k)

# Top level function for Singular Valued Decomposition
def SV_decomposition(k):
    U, E, Vt = SVD(ratings, k, option = 0)
    return U, np.diag(E), Vt

# Top Level function for CUR Decomposition
def CUR_decomposition(k):

    # CUR algorithm is written based on class slides
    m, n = ratings.shape

    ratings_squared = np.square(ratings)
    P_row = np.sum(ratings_squared, axis=1)
    P_col = np.sum(ratings_squared, axis=0)

    total_sum = np.sum(P_row)

    P_row = P_row / total_sum
    P_col = P_col / total_sum

    row_normalize = np.sqrt(k * P_row)
    col_normalize = np.sqrt(k * P_col)
    
    row_indeces = np.random.choice(m, k, p = P_row)
    col_indeces = np.random.choice(n, k, p = P_col)


    C = ratings[:, col_indeces]
    R = ratings[row_indeces, :]

    for i in range(k):
        C[:, i] = C[:, i] / col_normalize[col_indeces[i]]
        R[i, :] = R[i, :] / row_normalize[row_indeces[i]]

    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            W[i][j] = ratings[row_indeces[i]][col_indeces[j]]
    
    X, E, Yt = SVD(W, k, option = 1)
    for i in range(E.shape[0]):
        if E[i] > 0:
            E[i] = 1 / E[i]
    Y = np.transpose(Yt)
    E = np.diag(E)
    Xt = np.transpose(X)
    U = Y.dot(E.dot(Xt))

    return C, U, R

# It generates various metrics (storage, compute time and RMSE) for SVD and CUR
def generate_metrics():
    error_svd, error_cur = [], []
    space_svd, space_cur = [], []
    time_svd, time_cur = [], []

    K = range(10, 610, 10)
    for k in K:
        start1 = time.time()
        A1, B1, C1 = SV_decomposition(k)
        error1 = compute_RMSE(A1, B1, C1)
        end1 = time.time()

        # Computing the storage requirement for SVD
        storage1 = (A1.size * A1.itemsize) + (B1.size * B1.itemsize) + (C1.size * C1.itemsize)

        error2 = np.inf
        storage2 = np.inf

        start2 = time.time()

        # Since CUR is probabilistic algorithm, RMSE can be unexpected in one trial
        # As CUR is computationally fast, I am running three trials of CUR for given
        # k and picking the one with minimum RMSE.

        for _ in range(3):
            A2, B2, C2 = CUR_decomposition(k)
            error2_ = compute_RMSE(A2, B2, C2)

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

        end2 = time.time()

        error_svd.append(error1)
        error_cur.append(error2)

        space_svd.append(storage1)
        space_cur.append(storage2)
        
        time_svd.append(end1 - start1)
        time_cur.append(end2 - start2)
        if k % 50 == 0:
            print(f'===============  #Latent Factors = {k} =================')
            print(f"SVD Error = {error_svd[-1]:.4f}, SVD Time Taken = {time_svd[-1]:.4f}s, SVD Space Req. = {space_svd[-1]/1024/1024:.4f}MB")
            print(f"CUR Error = {error_cur[-1]:.4f}, CUR Time Taken = {time_cur[-1]:.4f}s, CUR Space Req. = {space_cur[-1]/1024/1024:.4f}MB")


    plot_metrics(K, error_svd, space_svd, time_svd, error_cur, space_cur, time_cur)

# Question 6
def PQ_Decomposition():

    # Computes Mean Squared Error
    def compute_error(P, Q):
        computed_ratings = P.dot(np.transpose(Q))
        train_error, test_error = 0, 0

        for i in range(train_set.shape[0]):
            rP = users_idx[train_set[i][0]]
            rQ = items_idx[train_set[i][1]]        
            train_error += np.square(ratings[rP][rQ] - computed_ratings[rP][rQ])
        train_error = train_error / train_set.shape[0]

        for i in range(test_set.shape[0]):
            rP = users_idx[test_set[i][0]]
            rQ = items_idx[test_set[i][1]]        
            test_error += np.square(ratings[rP][rQ] - computed_ratings[rP][rQ])
        test_error = test_error / test_set.shape[0]

        return train_error, test_error

    
    # regularization parameter
    L = 0.005

    # learning rate
    l = 0.001

    # Latent Factors
    k = 100

    m, n = ratings.shape
    P = np.random.random((m, k))
    Q = np.random.random((n, k))

    # Train Test splitting
    dataset = df[['userId', 'movieId']].values
    np.random.shuffle(dataset)
    train_set = dataset[:80670]
    test_set = dataset[80670:]

    train_errors, test_errors = [], []

    # Training Loop
    for epoch in range(35):

        # Stochastic Gradient Descent
        for i in range(train_set.shape[0]):
            rP = users_idx[train_set[i][0]]
            rQ = items_idx[train_set[i][1]]

            rating_exp = ratings[rP][rQ]
            rating_cal = P[rP].dot(Q[rQ])
            coeff = -2 * (rating_exp - rating_cal)

            dP = 2 * L * P[rP]
            dQ = 2 * L * Q[rQ]
            dP = dP + coeff * Q[rQ, :]
            dQ = dQ + coeff * P[rP, :]

            P[rP] = P[rP] - l * dP
            Q[rQ] = Q[rQ] - l * dQ
        
        train_error, test_error = compute_error(P, Q)
        train_errors.append(train_error)
        test_errors.append(test_error)

        print(f'#Epoch = {epoch}, Train Error = {train_error:.4f}, Test Error = {test_error:.4f}')
    
    # Plot
    plt.plot(train_errors, label = "Training Error")
    plt.plot(test_errors, label = "Test Error")
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title("PQ Decomposition")
    plt.legend()
    plt.show()


# Question 7

def generate_train_test_set():
    Y, _Y = [], []
    for i in range(M):
        for j in range(N):
            if ratings[i][j] > 0:
                Y.append([i, j])
            else:
                _Y.append([i, j])

    x_train, x_test, _, _ = train_test_split(np.array(Y), np.ones(len(Y)), test_size=0.3)
    _x_train, _x_test, _, _ = train_test_split(np.array(_Y), np.zeros(len(_Y)), test_size=x_test.shape[0] / len(_Y))

    y_test = np.concatenate([np.ones(x_test.shape[0]), np.zeros(_x_test.shape[0])])
    x_test = np.concatenate([x_test, _x_test])

    return x_train, _x_train, x_test, y_test


# Sigmoid Activation Function
def sigmoid(x, derivative = False):
    if derivative:
        return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
    else:
        return 1 / (1 + np.exp(-x))

# Multi Layer Perceptron
def MLP(K):
    input_layer0 = M
    input_layer1 = N
    embd_layer0 = K
    embd_layer1 = K
    NCF_layer1 = K
    NCF_layer2 = K
    output_layer = 1

    nnet = {
        'P': np.random.random((embd_layer0, input_layer0)),
        'Q': np.random.random((embd_layer1, input_layer1)),
        'w0': np.random.random((NCF_layer1, embd_layer0 + embd_layer0)),
        'w1': np.random.random((NCF_layer2, NCF_layer1)),
        'w2': np.random.random((output_layer, NCF_layer2))
    }
    return nnet

def forward_pass(user, item):
    
    # NN state: internal sums, neuron outputs
    nn_state = {}

    # User Latent Vector
    nn_state['user'] = user
    nn_state['item'] = item

    nn_state['ULV'] = np.dot(model['P'], user)

    # Item Latent Vector
    nn_state['ILV'] = np.dot(model['Q'], item)

    nn_state['z0'] = np.concatenate((nn_state['ULV'], nn_state['ILV']))
    nn_state['o0'] = nn_state['z0']

    # from input layer to hidden layer 1
    # weighted sum of all activations, then sigmoid
    nn_state['z1'] = np.dot(model['w0'], nn_state['o0'])
    nn_state['o1'] = sigmoid(nn_state['z1'])
    
    # from hidden 1 to hidden 2
    nn_state['z2'] = np.dot(model['w1'], nn_state['o1'])
    nn_state['o2'] = sigmoid(nn_state['z2'])
    
    # from hidden 2 to output
    nn_state['z3'] = np.dot(model['w2'], nn_state['o2'])
    nn_state['o3'] = sigmoid(nn_state['z3'])
    
    return nn_state


def backward_pass(user, item, y):
    # do the forward pass, register the state of the network
    nn_state = forward_pass(user, item)
    
    nn_state['d3'] = nn_state['o3'] - y
    nn_state['d2'] = np.dot(nn_state['d3'], model['w2']) * sigmoid(nn_state['z2'], derivative = True)
    nn_state['d1'] = np.dot(nn_state['d2'], model['w1']) * sigmoid(nn_state['z1'], derivative = True)

    nn_state['d0'] = np.dot(nn_state['d1'], model['w0']) * sigmoid(nn_state['z0'], derivative = True)

    # large deltas: adjustments to weights
    K = model['P'].shape[0]
    nn_state['D2'] = np.outer(nn_state['d3'], nn_state['o2'])
    nn_state['D1'] = np.outer(nn_state['d2'], nn_state['o1'])
    nn_state['D0'] = np.outer(nn_state['d1'], nn_state['o0'])
    
    nn_state['dP'] = np.outer(nn_state['d0'][:K], nn_state['user'])
    nn_state['dQ'] = np.outer(nn_state['d0'][K:], nn_state['item'])

    return nn_state

def compute_loss(x_train, y_train, x_test, y_test):
    user = np.zeros(M)
    item = np.zeros(N)
    test_loss, train_loss = 0, 0

    train_output = np.zeros(x_train.shape[0])
    test_output = np.zeros(x_test.shape[0])

    for i in range(x_train.shape[0]):

        user[x_train[i][0]] = 1
        item[x_train[i][1]] = 1

        train_output[i] = forward_pass(user, item)['o3'][0]

        user[x_train[i][0]] = 0
        item[x_train[i][1]] = 0
    
    train_loss = log_loss(y_train, train_output, labels = [0, 1])

    for i in range(x_test.shape[0]):

        user[x_test[i][0]] = 1
        item[x_test[i][1]] = 1

        test_output[i] = forward_pass(user, item)['o3'][0]
         

        user[x_test[i][0]] = 0
        item[x_test[i][1]] = 0
    
    test_loss = log_loss(y_test, test_output, labels = [0, 1])    
    return train_loss, test_loss

def train():

    epochs = 1
    lr = 0.01 # learning rate

    user = np.zeros(M)
    item = np.zeros(N)

    x_train1, _x_train, x_test, y_test = generate_train_test_set()
    y_train1 = np.ones(x_train1.shape[0])

    for e in range(epochs):
        start = time.time()

        _, x_train2, _, y_train2 = train_test_split(_x_train, np.zeros(_x_train.shape[0]), test_size = x_train1.shape[0] / _x_train.shape[0], random_state = 10)
        x_train, _, y_train, _ = train_test_split(np.concatenate([x_train1, x_train2]), np.concatenate([y_train1, y_train2]), test_size = 1, random_state = 10) 
         
        for i in range(x_train.shape[0]):
            
            user[x_train[i][0]] = 1
            item[x_train[i][1]] = 1

            m_state = backward_pass(user, item, y_train[i])

            user[x_train[i][0]] = 0
            item[x_train[i][1]] = 0
                    
            # update weights
            model['w0'] -= lr * m_state['D0']
            model['w1'] -= lr * m_state['D1']
            model['w2'] -= lr * m_state['D2']
            model['P']  -= lr * m_state['dP']
            model['Q']  -= lr * m_state['dQ']

        train_loss, test_loss = compute_loss(x_train, y_train, x_test, y_test)
        end = time.time()

        print(f"Epoch = {e}, Time Taken = {end - start:0.4f}s, Training Loss= {train_loss}, Test Loss = {test_loss}")


if __name__ == '__main__':
    df = pd.read_csv('../data/ml-latest-small/ratings.csv')
    ratings, users_idx, items_idx = get_ratings_matrix()
    M, N = ratings.shape

    print("++++++++++++ Question 1 to 5 +++++++++++++\n")
    generate_metrics()

    print("\n++++++++++++    Question 6    +++++++++++++\n")
    PQ_Decomposition()

    print("\n++++++++++++    Question 7    +++++++++++++\n")
    model = MLP(50)
    train()
