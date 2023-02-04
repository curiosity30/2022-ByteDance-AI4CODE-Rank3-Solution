import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.simplefilter('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)


def get_svd_fea(data, user_col, item_col, n_component=16):
    print('-- get_svd_fea')
    """协同过滤svd特征"""

    concat = data[[user_col, item_col]].copy()
    print(concat.shape)
    user_cnt = concat[user_col].max() + 1
    item_cnt = concat[item_col].max() + 1

    ### 1.构建交互稀疏矩阵
    co_data = np.ones(len(concat))
    suv = concat[user_col].values
    item = concat[item_col].values 
    rating = sparse.coo_matrix((co_data, (suv, item)))
    rating = (rating > 0) * 1.0

    ### 2.进行SVD分解 
    ## svd for user-song pairs
    # n_component = 16
    [u, s, vt] = svds(rating, k=n_component)
    # print(s[::-1])
    # s_song = np.diag(s[::-1])
    
    ### 3.生成SVD特征向量
    print('user svd...')
    user_topics = pd.DataFrame(u[:, ::-1])
    user_topics.columns = [f'user_component_{i}' for i in range(n_component)]
    user_topics[user_col] = range(user_cnt)
    user_topics.to_pickle('../temp_data/user_svd_embed.pkl')
    print('save ok!')

    print('item svd...')
    item_topics = pd.DataFrame(vt.transpose()[:, ::-1])
    item_topics.columns = [f'item_component_{i}' for i in range(n_component)]
    item_topics[item_col] = range(item_cnt)
    item_topics.to_pickle('../temp_data/video_svd_embed.pkl')
    print('save ok!')

    del concat, co_data, suv, item, rating
    gc.collect()


def make_user_embedding(train, video_embed_df, user_svd_embed_df, n_component=32):
    print('-- make_user_embedding')
    
    # user 历史正反馈视频集合
    inter_data = train[(train['is_like']==1)|(train['is_finish']==1)]
    user_seq_video_df = inter_data.groupby('userid')['videoid'].apply(list).reset_index()
    user_seq_video_df.columns = ['userid','userseq']
    print(user_seq_video_df.shape)
    print(user_seq_video_df['userid'].nunique())

    # 增加 embed-cols
    n_component = 32
    embed_cols = [f'item_component_{i}' for i in range(n_component)]
    user_embed = pd.DataFrame(np.zeros(shape=(user_seq_video_df.shape[0], n_component)))
    user_video_embed_df = user_seq_video_df[['userid']].copy()
    user_video_embed_df = pd.concat([user_video_embed_df, user_embed], axis=1)
    user_video_embed_df.columns = ['userid'] + embed_cols
    print(user_video_embed_df.shape)
    del user_embed; gc.collect()

    # user embedding = MEAN POOLING( 用户正反馈视频 embedding)
    for idx, row in tqdm(user_seq_video_df.iterrows()):
        user_videos_embed = video_embed_df[video_embed_df['videoid'].isin(row['userseq'])][embed_cols].mean(axis=0)
        user_video_embed_df.loc[user_video_embed_df['userid']==row['userid'], embed_cols] = np.array(user_videos_embed)
    print('ok')

    # 没有正反馈视频的用户
    user_set = set(train['userid'].unique().tolist())
    inter_user_set = set(user_seq_video_df['userid'].unique().tolist())
    not_in_user_lst = list(user_set - inter_user_set)
    del train; gc.collect()

    # 没有正反馈视频的用户embedding用svd embedding代替
    not_in_user_embed_mat = user_svd_embed_df[user_svd_embed_df['userid'].isin(not_in_user_lst)]
    print(not_in_user_embed_mat.shape)
    not_in_user_embed_mat.columns = ['userid'] + embed_cols
    user_video_embed_df = pd.concat([user_video_embed_df, not_in_user_embed_mat], axis=0)
    user_video_embed_df.sort_values(by=['userid'], inplace=True)
    user_video_embed_df.reset_index(drop=True, inplace=True)
    user_video_embed_df['userid'] = user_video_embed_df['userid'].astype('int32')
    print(user_video_embed_df.shape)

    return user_video_embed_df


def read_all_data(train_data_path, test_path):
    print('-- read_all_data')
    # 读取全部数据 train & test
    # train_data_path = '../temp_data/df_train_sp.pkl'
    # test_path = '../temp_data/df_test_sp.pkl'
    train = pd.read_pickle(train_data_path)
    print(train.shape)
    test = pd.read_pickle(test_path)
    print(test.shape)

    data = train.append(test)
    data.drop(['is_like','is_favourite','is_share','is_finish','tag'], axis=1, inplace=True)
    print('all data shape: ', data.shape)
    del train, test; gc.collect()

    data = data.reset_index(drop=True).reset_index()
    data.sort_values(by=['userid','videoid'], inplace=True)

    return data


def compute_similarity(user_group_df, user_embed_mat, video_embed_mat):
    print('-- compute_similarity')
    # 计算相似度：每个user embedding & 一组曝光video embedding 的余弦相似度
    row_num = user_group_df.shape[0]
    user_group_df['user_sim_lst'] = [[0] for i in range(row_num)]

    for i in tqdm(range(row_num)):
        user = user_group_df['userid'][i]
        vid_lst = user_group_df['user_vid_lst'][i]
        user_embed = user_embed_mat[user].reshape(1,-1)
        vid_embed = video_embed_mat[vid_lst]
        user_group_df['user_sim_lst'][i] = cosine_similarity(user_embed, vid_embed)[0]
    
    return user_group_df


def flatten_similarity_fea(user_group_df, data):
    print('-- flatten_similarity_fea')
    # 将相似度数组展平为一维数组，便于直接拼接到 data，作为特征
    mat = np.array(user_group_df['user_sim_lst'].tolist())
    data_sim_lst = np.zeros(data.shape[0])
    print(mat.shape)
    print(data_sim_lst.shape)

    start_idx = 0
    for i in tqdm(range(mat.shape[0])):
        arr_length = len(mat[i])
        end_idx = start_idx + arr_length
        data_sim_lst[start_idx: end_idx] = mat[i]
        start_idx += arr_length

    # 保存相似度特征
    data_sim_lst = data_sim_lst.astype('float32')
    
    return data_sim_lst


def generate_similarity_fea(n_component=32):
    
    print('================= Generate Similarity Feature ========================')
    # =============================== PARAMETERS ==================================
    train_data_path = '../temp_data/df_train_sp.pkl'
    test_path = '../temp_data/df_test_sp.pkl'
    video_embed_path = '../temp_data/video_svd_embed.pkl'
    user_svd_embed_path = '../temp_data/user_svd_embed.pkl'
    
    # ================== STEP 1: SVD分解得到video embedding ========================
    print('======= STEP 1: video embedding =======')
    train = pd.read_pickle(train_data_path)
    print(train.shape)

    get_svd_fea(data=train, user_col='userid', item_col='videoid', n_component=n_component)

    embed_cols = [f'item_component_{i}' for i in range(n_component)]
    user_svd_embed_df = pd.read_pickle(user_svd_embed_path)
    video_embed_df = pd.read_pickle(video_embed_path)
    video_embed_mat = video_embed_df[embed_cols].values.astype('float32')
    np.save('../temp_data/video_embed_mat.npy', video_embed_mat)
    
    # ================== STEP 2: 根据历史点击视频得到user embedding ===================
    print('======= STEP 2: user embedding =======')
    user_video_embed_df = make_user_embedding(train, video_embed_df, user_svd_embed_df, n_component=n_component)
    user_embed_mat = user_video_embed_df[embed_cols].values.astype('float32') # 保存为numpy格式
    np.save('../temp_data/user_embed_mat.npy', user_embed_mat)

    # =========================== STEP 3: 计算相似度 =================================
    print('======= STEP 3: compute similarity =======')
    data = read_all_data(train_data_path, test_path)
    # 用户、曝光视频列表
    user_group_df = data.groupby('userid')['videoid'].agg(list).reset_index()
    user_group_df.columns = ['userid','user_vid_lst']
    print(user_group_df.shape)
    # 计算相似度
    user_group_df = compute_similarity(user_group_df, user_embed_mat, video_embed_mat)
    data_sim_lst = flatten_similarity_fea(user_group_df, data)
    np.save('../temp_data/similarity_lst.npy', data_sim_lst)
    print('=== save similarity feature ok! ====')
    
    del data_sim_lst, data, user_group_df, user_video_embed_df, user_embed_mat, video_embed_df, video_embed_mat
    gc.collect()
    print(' Generate Similarity Feature END ')
    
