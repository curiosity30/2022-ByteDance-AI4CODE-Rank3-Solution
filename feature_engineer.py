import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, MinMaxScaler

from utils import *


def sparse_fea_encode(data, sparse_features):
    print('-- sparse_fea_encode...')
    # data[sparse_features] = data[sparse_features].fillna('-1', )
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    return data


def dense_fea_encode(data, dense_features):
    print('-- dense_fea_encode...')
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])
    for col in dense_features:
        data[col] = data[col].fillna(0, )
        qt = QuantileTransformer(n_quantiles=50, random_state=42)
        data[[col]] = qt.fit_transform(data[[col]])
    return data


def bin_encode(data, bin_cols, bin_num):
    print('-- bin_encode...')
    for col in bin_cols:
        data[col] = pd.qcut(data[col], q=bin_num, duplicates='drop')
        data[col] = data[col].cat.codes
    return data


# =================================== 通用统计函数 =============================================
def groupby_stat_func(df, groupby_cols, stat_cols, aggregations):
    """
    通用分组聚合函数
    """
    new_feas = []
    for stat_fea in stat_cols:
        # print(f'stat {stat_fea}')
        for group in groupby_cols:
            if isinstance(group, str): 
                group_name = group
            elif isinstance(group, list):
                group_name = '_'.join(group)
            
            for agg_type in aggregations:
                new_col_name = group_name + '_' + stat_fea + '_' + agg_type
                print(f'-- groupby stat {new_col_name}')
                df[new_col_name] = df.groupby(group)[stat_fea].transform(agg_type)
                new_feas.append(new_col_name)
    return df, new_feas


# =================================== 分箱特征 =============================================
def make_count_bins_fea(data, bin_flag=True):
    print('-- make_count_bins_fea')
    # quan_trans/qcut = 20/10
    cate_cols = ['userid','videoid','tag']
    new_feas = []
    for col in cate_cols:
        data[f'{col}_bin'] = data['videoid'].map(data['videoid'].value_counts())
        # data[f'{col}_bin'] = np.log1p(data[f'{col}_bin'])
        if bin_flag:
            qt = QuantileTransformer(n_quantiles=100, random_state=42)
            data[[f'{col}_bin']] = qt.fit_transform(data[[f'{col}_bin']])
            data[f'{col}_bin'] = pd.qcut(data[f'{col}_bin'], q=50, duplicates='drop')
            data[f'{col}_bin'] = data[f'{col}_bin'].cat.codes
        new_feas.append(f'{col}_bin')
    return data, new_feas


# =================================== 用户反馈特征（点赞收藏分享的统计） =============================================
def make_feedback_stat_fea(train, save_path, group_fea='videoid'):
    print(f'-- {group_fea}: make_feedback_stat_fea')

    stat_feas = []
    agg_lst = ['sum','mean']
    label_cols = ['is_like','is_favourite','is_share']

    stat_df = train.groupby([group_fea]).agg({
        'is_like': ['sum','mean'],
        'is_favourite': ['sum','mean'],
        'is_share': ['sum','mean'],
    }).reset_index()
    
    for label in label_cols:
        for agg in agg_lst:
            stat_feas += [f'{group_fea}_{label}_{agg}']
    
    stat_df.columns = [group_fea] + stat_feas
    # f'/home/workspace/output/stat_features/{group_fea}_stat.pkl'
    stat_df.to_pickle(save_path)
    print('save feature ok!')
    del stat_df; gc.collect()
    
    return


def merge_feedback_stat_fea(data, group_fea, save_path):
    print(f'-- {group_fea}: merge_feedback_stat_fea')
    
    # save_path = f'/home/workspace/output/stat_features/{group_fea}_stat.pkl'
    feedback_fea = pd.read_pickle(save_path)
    print('load feature ok!')
    data = data.merge(feedback_fea, how='left', on=group_fea)
    print('merge feature ok!')
    new_feas = [col for col in feedback_fea.columns if col != group_fea]
    del feedback_fea; gc.collect()
    
    return data, new_feas


# =================================== 用户对于tag反馈的统计特征 =============================================
def make_user_tag_stat_fea(train, save_path):
    print('-- make_user_tag_stat_fea')
    
    # user tag impression rate
    user_tag_stat_df = train.groupby(['userid','tag'])['videoid'].count().reset_index()
    user_tag_stat_df.columns = ['userid','tag','user_tag_cnt']
    count_df = train['userid'].value_counts().reset_index()
    count_df.columns = ['userid','userid_cnt']
    user_tag_stat_df = user_tag_stat_df.merge(count_df, how='left', on='userid')
    user_tag_stat_df['user_tag_ratio'] = user_tag_stat_df['user_tag_cnt'] / user_tag_stat_df['userid_cnt']
    
    # P( action | tag ): user 各类tag视频的点赞/收藏/分享 ratio
    user_tag_febk_df = train.groupby(['userid','tag']).agg({
        'is_like': ['mean','sum'],
        'is_favourite': ['mean','sum'],
        'is_share': ['mean','sum'],
    }).reset_index()
#     user_tag_febk_df.columns = ['userid','tag','tag_like_mean','tag_like_sum','tag_fav_mean','tag_fav_sum','tag_share_mean','tag_share_sum',]
    user_tag_febk_df.columns = ['userid','tag','like_m','like_s','fav_m','fav_s','share_m','share_s',]
    user_tag_stat_df = user_tag_stat_df.merge(user_tag_febk_df, how='left', on=['userid','tag'])
    
    # P( tag | action ): user 点赞/收藏/分享中的各个tag比例
    user_tag_stat_df['user_like_sum'] = user_tag_stat_df.groupby(['userid'])['like_s'].transform('sum')
    user_tag_stat_df['user_fav_sum'] = user_tag_stat_df.groupby(['userid'])['fav_s'].transform('sum')
    user_tag_stat_df['user_share_sum'] = user_tag_stat_df.groupby(['userid'])['share_s'].transform('sum')

    user_tag_stat_df['like_tag_rate'] = user_tag_stat_df['like_s'] / user_tag_stat_df['user_like_sum']
    user_tag_stat_df['fav_tag_rate'] = user_tag_stat_df['fav_s'] / user_tag_stat_df['user_fav_sum']
    user_tag_stat_df['share_tag_rate'] = user_tag_stat_df['share_s'] / user_tag_stat_df['user_share_sum']

    user_tag_stat_df.drop(['user_like_sum','user_fav_sum','user_share_sum'], axis=1, inplace=True)
    user_tag_stat_df.drop(['like_s','fav_s','share_s','user_tag_cnt','userid_cnt'], axis=1, inplace=True)
    
    # save file
    # '/home/workspace/output/stat_features/user_tag_stat.pkl'
    user_tag_stat_df.to_pickle(save_path)
    print('save feature ok!')
    
    del user_tag_stat_df; gc.collect()
    
    return 

def merge_user_tag_stat_fea(data, save_path):
    print('-- merge_user_tag_stat_fea')
    
    # save_path = '/home/workspace/output/stat_features/user_tag_stat.pkl'
    user_tag_stat_df = pd.read_pickle(save_path)
    print('load feature ok!')
    data = data.merge(user_tag_stat_df, how='left', on=['userid','tag'])
    print('merge feature ok!')
    
    new_feas = [col for col in user_tag_stat_df.columns if col not in ['userid','tag']]
    del user_tag_stat_df; gc.collect()
    
    return data, new_feas


# ===================================ALL Feature Engineer=========================================

def make_feature_engineer(data, sparse_features, dense_features):
    # 相似性特征
    print('-- merge similarity fea...')
    # sim_fea_path = '/home/workspace/output/embedding/similarity_lst.npy'
    sim_fea_path = '../temp_data/similarity_lst.npy'
    data_sim_lst = np.load(sim_fea_path)
    data['user_video_sim'] = data_sim_lst
    data = reduce_mem_usage(data)
    dense_features += ['user_video_sim',]
    print(data.shape)

    # KMEANS FEATURE
    print('-- merge kmeans feature...')
    user_kmeans_path = '../temp_data/user_kmeans_label.csv'
    video_kmeans_path = '../temp_data/video_kmeans_label.csv'
    user_kmeans_fea = pd.read_csv(user_kmeans_path)
    video_kmeans_fea = pd.read_csv(video_kmeans_path)
    data = data.merge(user_kmeans_fea, how='left', on=['userid'])
    print(data.shape)
    data = data.merge(video_kmeans_fea, how='left', on=['videoid'])
    print(data.shape)
    sparse_features += ['user_kmeans_label', 'video_kmeans_label']

    # user-tag 交叉特征
    print('-- make user_tag...')
    data['user_tag'] = data['userid'].astype(str) + '_' + data['tag'].astype(str)
    sparse_features += ['user_tag',]
    data = sparse_fea_encode(data, sparse_features=['user_tag',])

    # 分箱特征
    data, bin_feas = make_count_bins_fea(data, bin_flag=False)
    sparse_features += bin_feas

    # user对于tag的反馈统计特征
    usertag_path = '../temp_data/user_tag_stat_v1.pkl'
    data, usertag_feas = merge_user_tag_stat_fea(data, save_path=usertag_path)
    dense_features += usertag_feas

    # video的反馈特征
    video_fb_path = '../temp_data/video_stat_v1.pkl'
    data, video_feas = merge_feedback_stat_fea(data, group_fea='videoid', save_path=video_fb_path)
    drop_cols = ['videoid_is_like_mean','videoid_is_favourite_mean','videoid_is_share_mean']
    data.drop(drop_cols, axis=1, inplace=True)
    video_feas = [col for col in video_feas if col not in drop_cols]
    dense_features += video_feas

    # user与tag的交叉统计
    data, usertag_nuq_feas = groupby_stat_func(data, groupby_cols=[['userid','tag']], stat_cols=['videoid'], aggregations=['nunique'])
    dense_features += usertag_nuq_feas
    data = reduce_mem_usage(data)
    
    return data, sparse_features, dense_features