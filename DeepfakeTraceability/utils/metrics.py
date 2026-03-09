import logging
from collections import Counter

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils.reranking import re_ranking

logger = logging.getLogger(__name__)


def parse_metric_for_print(best_metrics):
    if best_metrics is None:
        return "\n"
    max_key_len = max(len(str(key)) for key in best_metrics.keys()) + 2
    max_val_len = max(len(str(value)) for value in best_metrics.values()) + 2

    content_lines = []
    for key, value in best_metrics.items():
        line = f"| {key:<{max_key_len}} | {value:<{max_val_len}} |"
        content_lines.append(line)
    max_line_length = max(len(line) for line in content_lines)
    result = ""
    result += "=" * max_line_length + "\n"
    result += f"| {'Best metric':<{max_key_len + max_val_len + 3}} |\n"
    result += "=" * max_line_length + "\n"
    for line in content_lines:
        result += line + "\n"
    result += "=" * max_line_length
    return result


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_df_ids, g_df_ids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_df_ids[indices] == q_df_ids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid
        q_df_id = q_df_ids[q_idx]

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, use_pca=True,
                 pca_dim=128, use_clustering=True, n_clusters=500, use_medoid=True):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters
        self.use_medoid = use_medoid

    def reset(self):
        self.feats = []
        self.df_ids = []

    def update(self, output):  # called once for each batch
        feat, df_id = output
        self.feats.append(feat.cpu())
        self.df_ids.extend(np.asarray(df_id.cpu().numpy()))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)

        # 0. 初始归一化 (可选，通常 PCA 前不需要强归一化，但为了稳定性可以保留)
        # if self.feat_norm:
        #    feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # 分离 Query 和 Gallery
        qf = feats[:self.num_query]
        q_df_ids = np.asarray(self.df_ids[:self.num_query])
        gf = feats[self.num_query:]
        g_df_ids = np.asarray(self.df_ids[self.num_query:])

        # ======================================================
        # 1. PCA 降维 (挖掘共性特征)
        # ======================================================
        if self.use_pca:
            logger.info(f'=> Applying PCA: reducing dim from {gf.shape[1]} to {self.pca_dim}...')

            # 转为 numpy 进行 sklearn 处理
            gf_np = gf.numpy()
            qf_np = qf.numpy()

            # 在 Gallery 上拟合 PCA (假设 Gallery 代表了数据分布)
            pca = PCA(n_components=self.pca_dim, random_state=42)
            gf_np = pca.fit_transform(gf_np)
            # 将变换应用到 Query
            qf_np = pca.transform(qf_np)

            # 转回 Tensor
            gf = torch.from_numpy(gf_np).float()
            qf = torch.from_numpy(qf_np).float()

            logger.info(f'=> PCA done. Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}')

        # 降维后再次归一化 (关键步骤！降维破坏了单位球面的性质，必须重新归一化以便计算余弦距离)
        if self.feat_norm:
            qf = torch.nn.functional.normalize(qf, dim=1, p=2)
            gf = torch.nn.functional.normalize(gf, dim=1, p=2)

        # # ======================================================
        # # 2. 聚类 (基于降维后的特征)
        # # ======================================================
        if self.use_clustering:
            logger.info(f'=> Clustering Gallery into {self.n_clusters} clusters (Medoid={self.use_medoid})...')

            gf_np = gf.numpy()

            # 使用 KMeans 聚类
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10).fit(gf_np)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            new_gf_list = []
            new_g_df_ids = []

            for i in range(self.n_clusters):
                indices = np.where(labels == i)[0]
                if len(indices) == 0: continue

                # A. 确定 Label (多数投票)
                original_ids = g_df_ids[indices]
                most_common_id = Counter(original_ids).most_common(1)[0][0]
                new_g_df_ids.append(most_common_id)

                # B. 确定 Feature (Medoid 或 Centroid)
                if self.use_medoid:
                    # 找离中心最近的真实样本
                    cluster_samples = gf_np[indices]
                    center_vec = centers[i]
                    dists = np.linalg.norm(cluster_samples - center_vec, axis=1)
                    best_sample = cluster_samples[np.argmin(dists)]
                    new_gf_list.append(best_sample)
                else:
                    new_gf_list.append(centers[i])

            # 更新 Gallery
            gf = torch.from_numpy(np.array(new_gf_list)).float()
            g_df_ids = np.asarray(new_g_df_ids)

            # 聚类中心/Medoid 也需要重新归一化
            if self.feat_norm:
                gf = torch.nn.functional.normalize(gf, dim=1, p=2)

            logger.info(f'=> Gallery clustered. New size: {gf.shape[0]}')
        #
        # # ======================================================
        # # 3. 匹配 (Query -> Processed Gallery)
        # # ======================================================
        if self.reranking:
            logger.info('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            logger.info('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        cmc, mAP = eval_func(distmat, q_df_ids, g_df_ids)

        return cmc, mAP, distmat, self.df_ids, qf, gf
