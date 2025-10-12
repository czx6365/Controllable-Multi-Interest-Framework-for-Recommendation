
import random

class DataIterator:
    """数据迭代器类，用于处理用户-物品交互数据，生成模型训练和评估所需的批次数据"""

    def __init__(self, source,
                 batch_size=128,#批次大小，batch_size=1 一次训练完所有
                 maxlen=100,
                 train_flag=0
                 ):
        # 读取数据源
        self.read(source)
        # 将用户集合转换为列表
        self.users = list(self.users)

        self.batch_size = batch_size  # 训练批次大小
        self.eval_batch_size = batch_size  # 评估批次大小
        self.train_flag = train_flag  # 训练/评估模式标记，0表示训练，非0表示评估
        self.maxlen = maxlen  # 历史序列的最大长度
        self.index = 0  # 用于评估时记录用户索引位置

    def __iter__(self):
        """使类成为可迭代对象"""
        return self

    def next(self):
        return self.__next__()

    def read(self, source):
        """
        读取并解析数据源文件
        参数:
            source: 数据源文件路径
        """
        self.graph = {}  # 存储用户的物品交互序列，键为用户ID，值为物品ID列表（按时间排序）
        self.users = set()  # 存储所有用户ID
        self.items = set()  # 存储所有物品ID

        with open(source, 'r') as f:
            for line in f:
                # 解析每行数据，格式为"用户ID,物品ID,时间戳"
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                time_stamp = int(conts[2])

                # 将用户和物品添加到集合中
                self.users.add(user_id)
                self.items.add(item_id)

                # 构建用户的物品交互序列
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((item_id, time_stamp))

        # 对每个用户的交互序列按时间戳排序，并只保留物品ID
        for user_id, value in self.graph.items():
            # 按时间戳排序
            value.sort(key=lambda x: x[1])
            # 提取物品ID，形成最终的用户交互序列
            self.graph[user_id] = [x[0] for x in value]

        # 将用户和物品集合转换为列表
        self.users = list(self.users)
        self.items = list(self.items)

    def __next__(self):
        """
        生成下一个批次的数据
        返回:
            (用户ID列表, 目标物品ID列表), (历史物品列表, 历史掩码列表)
        """
        # 训练模式：随机采样batch_size个用户
        if self.train_flag == 0:
            user_id_list = random.sample(self.users, self.batch_size)
        # 评估模式：按顺序取用户，循环迭代
        else:
            total_user = len(self.users)
            # 如果索引超出用户总数，重置索引并停止迭代
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            # 取从当前索引开始的eval_batch_size个用户
            user_id_list = self.users[self.index: self.index + self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []  # 存储目标物品ID
        hist_item_list = []  # 存储历史物品序列
        hist_mask_list = []  # 存储历史序列的掩码（用于标识有效元素）

        for user_id in user_id_list:
            # 获取该用户的物品交互序列
            item_list = self.graph[user_id]

            # 训练模式：随机选择一个位置作为目标物品
            if self.train_flag == 0:
                # 从第4个位置到序列末尾随机选择一个位置k
                k = random.choice(range(4, len(item_list)))
                # 将位置k的物品作为目标物品
                item_id_list.append(item_list[k])
            # 评估模式：使用序列后20%的物品作为目标物品
            else:
                # 取序列长度的80%位置作为分割点
                k = int(len(item_list) * 0.8)
                # 将位置k之后的物品作为目标物品
                item_id_list.append(item_list[k:])

            # 处理历史物品序列，使其长度为maxlen
            if k >= self.maxlen:
                # 如果历史序列长度超过maxlen，取最近的maxlen个物品
                hist_item_list.append(item_list[k - self.maxlen: k])
                hist_mask_list.append([1.0] * self.maxlen)  # 掩码全为1（全部有效）
            else:
                # 如果历史序列长度不足maxlen，用0填充，并设置相应的掩码
                hist_item_list.append(item_list[:k] + [0] * (self.maxlen - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.maxlen - k))  # 有效位置为1，填充位置为0

        return (user_id_list, item_id_list), (hist_item_list, hist_mask_list)