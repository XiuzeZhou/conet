import numpy as np
from scipy import sparse


# read original records
def load_data(file_dir):
    # output: 
    # N: the number of user;
    # M: the number of item
    # data: the list of rating information
    user_ids_dict, rated_item_ids_dict = {},{}
    N, M, u_idx, i_idx = 0,0,0,0 
    data_rating = []
    data_time = []
    f = open(file_dir)
    for line in f.readlines():
        if '::' in line:
            u, i, r = line.split('::')[:3]
        elif ',' in line:
            u, i, r = line.split(',')[:3]
        else:
            u, i, r = line.split()[:3]
    
        if u not in user_ids_dict:
            user_ids_dict[u]=u_idx
            u_idx+=1
        if i not in rated_item_ids_dict:
            rated_item_ids_dict[i]=i_idx
            i_idx+=1
        data_rating.append([user_ids_dict[u],rated_item_ids_dict[i],float(r)])
    
    f.close()
    N = u_idx
    M = i_idx

    return N, M, data_rating


def sequence2mat(sequence, N, M, is_sparse=False):
    # input:
    # sequence: the list of rating information
    # N: row number, i.e. the number of users
    # M: column number, i.e. the number of items
    # output:
    # mat: user-item rating matrix
    records_array = np.array(sequence)
    row = records_array[:,0].astype(int)
    col = records_array[:,1].astype(int)
    values = records_array[:,2].astype(np.float32)
    if is_sparse:
        mat = sparse.csr_matrix((values, (row, col)), shape=(N, M))
    else:
        mat = np.zeros((N,M))
        mat[row,col] = values
    
    return mat


# Read train and test data
def read_data(file_dir):
    data=[]
    f = open(file_dir)
    for line in f.readlines():
        if len(line.split())==4:
            u, i, r, _ = line.split()
            data.append([int(u), int(i), float(r)])
        else:
            data.append([int(i) for i in line.split()[1:]])
    
    return data


def generate_data(train_mat, positive_size=1, list_length=5, sample_size=2):
    data = []
    users_num,items_num = train_mat.shape
    for u in range(users_num):
        rated_items = np.where(train_mat[u,:]>0)[0] #用户u中有评分项的id
        
        for _ in range(sample_size):
            for item0 in rated_items:
                line = []
                line.append([u,item0,1])
                i = 1 # 正样本数
                while i < positive_size:
                    item1 = np.random.randint(items_num)
                    if (item1 in rated_items) and (item1 !=item0):
                        line.append([u,item1,1])
                        i = i+1
                i = 0 # 负样本数
                while i<list_length-positive_size:
                    item1 = np.random.randint(items_num)
                    if item1 not in rated_items:
                        line.append([u,item1,0])
                        i = i+1
                np.random.shuffle(line) 
                data.append(np.array(line))
    return data


def get_train_test(rating_mat):
    n,m = rating_mat.shape
    
    selected_items,rest_ratings,negtive_items = [],[],[]
    for user_line in rating_mat:
        rated_items = np.where(user_line>0)[0]
        rated_num = len(rated_items)
        random_ids = [i for i in range(rated_num)]
        np.random.shuffle(random_ids)
        selected_id = random_ids[0]
        selected_items.append(rated_items[selected_id])
        rest_ratings.append(rated_items[random_ids[1:]])
        
        unrated_items = np.where(user_line==0)[0]
        unrated_num = len(unrated_items)
        random_ids = [i for i in range(unrated_num)]
        np.random.shuffle(random_ids)
        negtive_items.append(unrated_items[random_ids[:99]])
        
    train = [[user, item, rating_mat[user,item]] for user in range(n) for item in rest_ratings[user]]   
    test = [[user, selected_items[user]] for user in range(n)]
    
    np.random.shuffle(train)  
    return train,test,negtive_items