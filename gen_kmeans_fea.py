import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def generate_kmeans_fea(n_component, user_cluster_num, video_cluster_num):
    
    print('================= Generate Kmeans Feature ========================')
    # n_component = 32
    # USER_CLUSTER_NUM = 16
    # VIDEO_CLUSTER_NUM = 48
    embed_cols = [f'item_component_{i}' for i in range(n_component)]

    # USER EMBED MATRIX
    user_embed_path = '../temp_data/user_embed_mat.npy'
    user_embed_mat = np.load(user_embed_path)
    print(user_embed_mat.shape)
    user_df = pd.DataFrame(user_embed_mat)
    user_df.columns = embed_cols
    user_df['userid'] = list(range(user_df.shape[0]))

    # VIDEO EMBED MATRIX
    video_embed_path = '../temp_data/video_embed_mat.npy'
    video_embed_mat = np.load(video_embed_path)
    print(video_embed_mat.shape)
    vid_df = pd.DataFrame(video_embed_mat)
    vid_df.columns = embed_cols
    vid_df['videoid'] = list(range(vid_df.shape[0]))

    # USER KMEANS
    print('user kmeans model fitting...')
    user_kmeans_path = '../temp_data/user_kmeans_label.csv'
    cluster_model = KMeans(n_clusters=user_cluster_num, random_state=42)
    cluster_model.fit(user_df[embed_cols].values)
    user_df['user_kmeans_label'] = cluster_model.predict(user_df[embed_cols].values)
    user_df[['userid','user_kmeans_label']].to_csv(user_kmeans_path, index=False)
    print('save user_kmeans_label ok!')

    # VIDEO KMEANS
    print('video kmeans model fitting...')
    video_kmeans_path = '../temp_data/video_kmeans_label.csv'
    cluster_model = KMeans(n_clusters=video_cluster_num, random_state=42)
    cluster_model.fit(vid_df[embed_cols].values)
    vid_df['video_kmeans_label'] = cluster_model.predict(vid_df[embed_cols].values)
    vid_df[['videoid','video_kmeans_label']].to_csv(video_kmeans_path, index=False)
    print('save video_kmeans_label ok!')
    print(' Generate Kmeans Feature END ')
