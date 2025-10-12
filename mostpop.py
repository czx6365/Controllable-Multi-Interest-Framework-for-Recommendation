import math
from collections import defaultdict


class MostPopular:
    """最受欢迎物品推荐算法类
    基于基于物品在训练集中的出现频率，向用户推荐最受欢迎的物品
    """

    def __init__(self, source):
        """初始化函数，读取训练集和测试集数据

        参数:
            source: 数据文件的路径前缀
        """
        # 读取训练集数据，构建用户-物品交互图
        self.train_graph = self.read(source + '_train.txt')
        # 读取测试集数据，构建用户-物品交互图
        self.test_graph = self.read(source + '_test.txt')

    def read(self, source):
        """读取数据文件，构建用户-物品交互图

        参数:
            source: 数据文件路径

        返回:
            graph: 字典，键为用户ID，值为该用户交互过的物品ID列表
        """
        graph = {}
        with open(source, 'r') as f:
            for line in f:
                # 分割每行数据，假设格式为"user_id,item_id,..."
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                # 如果用户ID不在图中，则初始化一个空列表
                if user_id not in graph:
                    graph[user_id] = []
                # 将物品添加到用户的交互列表中
                graph[user_id].append(item_id)
        return graph

    def evaluate(self, N=50):
        """评估推荐算法性能，计算召回率、NDCG和命中率

        参数:
            N: 推荐的物品数量

        返回:
            包含recall(召回率)、ndcg和hitrate(命中率)的字典
        """
        # 统计每个物品在训练集中的出现次数（ popularity）
        item_count = defaultdict(int)
        for user in self.train_graph.keys():
            for item in self.train_graph[user]:
                item_count[item] += 1

        # 按物品出现次数降序排序
        item_list = list(item_count.items())
        item_list.sort(key=lambda x: x[1], reverse=True)

        # 选取前N个最受欢迎的物品作为推荐列表
        item_pop = set()
        for i in range(N):
            # 防止物品数量不足N的情况
            if i < len(item_list):
                item_pop.add(item_list[i][0])

        # 初始化评估指标
        total_recall = 0.0  # 总召回率
        total_ndcg = 0.0  # 总NDCG
        total_hitrate = 0  # 总命中数

        # 遍历测试集中的每个用户
        for user in self.test_graph.keys():
            recall = 0  # 当前用户的召回数
            dcg = 0.0  # 当前用户的DCG值

            item_list = self.test_graph[user]


            # 取测试集物品列表的后20%作为评估目标（假设前80%是历史，后20%是待预测）
            item_list = item_list[int(len(item_list) * 0.8):]
            if not item_list:  # 避免空列表
                continue

            # 计算该用户的召回率和DCG
            for no, item_id in enumerate(item_list):
                # 如果物品在推荐列表中
                if item_id in item_pop:
                    recall += 1
                    # 计算DCG，位置越靠前权重越高
                    dcg += 1.0 / math.log(no + 2, 2)

            # 计算理想DCG（IDCG）
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no + 2, 2)

            # 累加总召回率
            total_recall += recall * 1.0 / len(item_list)

            # 计算并累加NDCG（如果有命中）
            if recall > 0:
                total_ndcg += dcg / idcg
                total_hitrate += 1  # 记录命中用户数

        # 计算平均指标
        total = len(self.test_graph)
        recall = total_recall / total
        ndcg = total_ndcg / total
        hitrate = total_hitrate * 1.0 / total

        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}


if __name__ == "__main__":
    # 初始化模型，读取书籍数据
    data = MostPopular(r'F:\科研\ComiRec-master\ComiRec-master\data\book_data\book')
    # 也可以选择读取淘宝数据
    # data = MostPopular('./data/taobao_data/taobao')

    # 评估并打印结果（推荐50个物品）
    print(data.evaluate(50))