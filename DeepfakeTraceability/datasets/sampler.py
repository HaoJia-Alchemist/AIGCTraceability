import math

from accelerate import Accelerator
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomDeepfakeSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances, accelerator):
        super().__init__()
        accelerator = Accelerator()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_df_ids_per_batch = self.batch_size // self.num_instances
        self.accelerator = accelerator
        # 分布式信息：获取当前进程的rank和总进程数
        self.rank = accelerator.process_index  # 当前进程编号（0-based）
        self.world_size = accelerator.num_processes  # 总进程数

        # 构建身份到索引的映射（不变）
        self.index_dic = defaultdict(list)
        for index, (_, df_id, _) in enumerate(self.data_source):
            self.index_dic[df_id].append(index)
        self.df_ids = list(self.index_dic.keys())

        # 计算全局总长度（所有进程的样本总和）
        self.global_length = 0
        for df_id in self.df_ids:
            idxs = self.index_dic[df_id]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.global_length += num - num % self.num_instances

        # 计算当前进程的本地长度（全局长度 / 总进程数，确保整除）
        self.length = self.global_length // self.world_size

    def __iter__(self):
        # 1. 确保每个进程的随机状态不同（基于Accelerator的种子偏移）
        # 复用Accelerator为当前进程设置的随机状态，无需手动seed
        # （Accelerator已自动为每个进程设置了不同的random/np/torch种子）

        # 2. 生成全局采样索引（与原始逻辑一致，但随机操作已绑定到当前进程）
        batch_idxs_dict = defaultdict(list)
        for df_id in self.df_ids:
            idxs = copy.deepcopy(self.index_dic[df_id])
            if len(idxs) < self.num_instances:
                # 使用当前进程的np随机状态（避免所有进程重复）
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            # 使用当前进程的random随机状态
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[df_id].append(batch_idxs)
                    batch_idxs = []

        avai_df_ids = copy.deepcopy(self.df_ids)
        global_idxs = []  # 全局所有进程的索引序列

        while len(avai_df_ids) >= self.num_df_ids_per_batch:
            selected_df_ids = random.sample(avai_df_ids, self.num_df_ids_per_batch)
            for df_id in selected_df_ids:
                batch_idxs = batch_idxs_dict[df_id].pop(0)
                global_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[df_id]) == 0:
                    avai_df_ids.remove(df_id)

        # 3. 将全局索引按进程分割：每个进程只取属于自己的部分
        # 确保每个进程的索引不重叠（按rank分配）
        local_idxs = global_idxs[self.rank::self.world_size]

        # 4. 截断到当前进程的长度（避免最后一个进程多出来的样本）
        return iter(local_idxs[:self.length])

    def __len__(self):
        return self.length