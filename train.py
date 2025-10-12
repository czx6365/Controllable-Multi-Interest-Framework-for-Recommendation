import math
import random
import shutil
import sys
import time
from collections import defaultdict
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import faiss
from data_iterator import DataIterator
from model import *



best_metric = 0


def prepare_data(src, target):
    """准备数据"""
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask


def load_item_cate(source):
    """加载物品类别映射表"""
    item_cate = {}
    try:
        with open(source, 'r', encoding='utf-8') as f:
            for line in f:
                conts = line.strip().split(',')
                if len(conts) >= 2:
                    item_id = int(conts[0])
                    cate_id = int(conts[1])
                    item_cate[item_id] = cate_id
    except Exception as e:
        print(f"加载类别文件时出错: {e}")
    return item_cate


def compute_diversity(item_list, item_cate_map):
    """计算推荐列表的多样性"""
    n = len(item_list)
    if n <= 1:
        return 0.0

    diversity = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if item_list[i] in item_cate_map and item_list[j] in item_cate_map:
                diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]

    diversity /= ((n - 1) * n / 2)
    return diversity


def create_faiss_index(item_embs, embedding_dim):
    """创建FAISS索引"""
    if faiss is None:
        print("警告: 使用numpy进行相似度搜索，性能可能较低")
        return None

    try:
        if tf.config.list_physical_devices('GPU'):
            # GPU版本
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0
            index = faiss.GpuIndexFlatIP(res, embedding_dim, flat_config)
        else:
            # CPU版本
            index = faiss.IndexFlatIP(embedding_dim)

        index.add(item_embs)
        return index
    except Exception as e:
        print(f"创建FAISS索引失败: {e}")
        return None


def numpy_search(user_embs, item_embs, topN):
    """使用numpy进行相似度搜索（FAISS的备选方案）"""
    similarities = np.dot(user_embs, item_embs.T)
    indices = np.argsort(similarities, axis=1)[:, ::-1][:, :topN]
    distances = np.take_along_axis(similarities, indices, axis=1)
    return distances, indices


def evaluate_full(model, test_data, model_path, batch_size, item_cate_map, save=True, coef=None):
    """全面评估模型性能"""
    topN = args.topN

    # 获取物品嵌入向量
    item_embs = model.output_item()

    # 创建搜索索引
    search_index = create_faiss_index(item_embs, args.embedding_dim)

    # 初始化评估指标
    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0

    for src, tgt in test_data:
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)
        user_embs = model.output_user([hist_item, hist_mask])

        if search_index is not None:
            # 使用FAISS搜索
            D, I = search_index.search(user_embs, topN)
        else:
            # 使用numpy搜索
            D, I = numpy_search(user_embs, item_embs, topN)

        if len(user_embs.shape) == 2:
            # 单兴趣模型
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)

                for no, iid in enumerate(I[i]):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)

                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)

                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(I[i], item_cate_map)
        else:
            # 多兴趣模型
            ni = user_embs.shape[1]
            user_embs_flat = np.reshape(user_embs, [-1, user_embs.shape[-1]])

            if search_index is not None:
                D, I = search_index.search(user_embs_flat, topN)
            else:
                D, I = numpy_search(user_embs_flat, item_embs, topN)

            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []

                if coef is None:
                    item_list = list(zip(
                        np.reshape(I[i * ni:(i + 1) * ni], -1),
                        np.reshape(D[i * ni:(i + 1) * ni], -1)
                    ))
                    item_list.sort(key=lambda x: x[1], reverse=True)

                    for j in range(len(item_list)):
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            item_cor_list.append(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else:
                    origin_item_list = list(zip(
                        np.reshape(I[i * ni:(i + 1) * ni], -1),
                        np.reshape(D[i * ni:(i + 1) * ni], -1)
                    ))
                    origin_item_list.sort(key=lambda x: x[1], reverse=True)

                    item_list = []
                    tmp_item_set = set()
                    for (x, y) in origin_item_list:
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)

                    cate_dict = defaultdict(int)
                    for j in range(topN):
                        if not item_list:
                            break

                        max_index = 0
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)):
                            current_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            if current_score > max_score:
                                max_index = k
                                max_score = current_score

                        item_list_set.add(item_list[max_index][0])
                        item_cor_list.append(item_list[max_index][0])
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)

                true_item_set = set(iid_list)
                for no, iid in enumerate(item_cor_list):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)

                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)

                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)

        total += len(item_id)

    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total if not save else 0.0

    if save:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}


def get_model(dataset, model_type, item_count, batch_size, maxlen):
    """根据模型类型创建对应的模型实例"""
    model_classes = {
        'DNN': Model_DNN,
        'GRU4REC': Model_GRU4REC,
        'MIND': Model_MIND,
        'ComiRec-DR': Model_ComiRec_DR,
        'ComiRec-SA': Model_ComiRec_SA
    }

    if model_type not in model_classes:
        print(f"无效的模型类型: {model_type}")
        return None

    # 创建模型实例
    if model_type == 'MIND':
        relu_layer = dataset == 'book'
        model = model_classes[model_type](
            item_count, args.embedding_dim, args.hidden_size, batch_size,
            args.num_interest, maxlen, relu_layer=relu_layer
        )
    elif model_type in ['ComiRec-DR', 'ComiRec-SA']:
        model = model_classes[model_type](
            item_count, args.embedding_dim, args.hidden_size, batch_size,
            args.num_interest, maxlen
        )
    else:
        model = model_classes[model_type](
            item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen
        )

    # 构建模型
    print(f"构建 {model_type} 模型...")
    try:
        # 定义输入形状并构建模型
        input_shapes = [
            (None,),  # nick_id
            (None,),  # item_id
            (None, maxlen),  # hist_item
            (None, maxlen)  # hist_mask
        ]
        model.build(input_shapes)
        print("模型构建成功")
    except Exception as e:
        print(f"模型构建警告: {e}")
        # 尝试通过前向传播构建
        try:
            import numpy as np
            dummy_inputs = [
                np.zeros((batch_size,), dtype=np.int32),  # nick_id
                np.zeros((batch_size,), dtype=np.int32),  # item_id
                np.zeros((batch_size, maxlen), dtype=np.int32),  # hist_item
                np.zeros((batch_size, maxlen), dtype=np.int32)  # hist_mask
            ]
            _ = model(dummy_inputs, training=False)
            print("通过前向传播构建模型成功")
        except Exception as e2:
            print(f"模型构建失败: {e2}")
            return None

    return model


def get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=True):
    """生成实验名称"""
    extr_name = input('请输入实验名称: ')
    para_name = '_'.join([
        dataset, model_type, f'b{batch_size}',
        f'lr{lr}', f'd{args.embedding_dim}', f'len{maxlen}'
    ])
    exp_name = para_name + '_' + extr_name

    if save and os.path.exists('runs/' + exp_name):
        flag = input('实验名称已存在，是否覆盖? (y/n) ')
        if flag.lower() == 'y':
            shutil.rmtree('runs/' + exp_name)
        else:
            extr_name = input('请输入新的实验名称: ')
            exp_name = para_name + '_' + extr_name

    return exp_name


def setup_gpu():
    """设置GPU配置"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def train_model(
        train_file,
        valid_file,
        test_file,
        cate_file,
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        test_iter=50,
        model_type='DNN',
        lr=0.001,
        max_iter=100,
        patience=20
):
    """训练模型的主函数 - TensorFlow 2.x 版本"""
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen)
    best_model_path = "best_model/" + exp_name + '/'

    setup_gpu()

    # 初始化TensorBoard
    writer = None
    try:
        from tensorboard import SummaryWriter
        writer = SummaryWriter('runs/' + exp_name)
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter('runs/' + exp_name)
        except ImportError:
            print("警告: 未找到tensorboard库，将无法记录训练日志")
            writer = None

    item_cate_map = load_item_cate(cate_file)

    # 创建数据迭代器
    train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
    valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1)

    # 创建模型 - 不需要 session
    model = get_model(dataset, model_type, item_count, batch_size, maxlen)
    if model is None:
        return

    print('开始训练')
    start_time = time.time()
    iter_count = 0
    loss_sum = 0.0
    trials = 0
    global best_metric

    try:
        for src, tgt in train_data:
            nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)

            # 使用模型的 train_step 方法
            loss_dict = model.train_step([nick_id, item_id, hist_item, hist_mask, lr])
            loss = loss_dict['loss']

            loss_sum += loss
            iter_count += 1

            if iter_count % test_iter == 0:
                # 评估模型
                metrics = evaluate_full(model, valid_data, best_model_path, batch_size, item_cate_map)

                log_str = f'迭代: {iter_count}, 训练损失: {loss_sum / test_iter:.4f}'
                if metrics:
                    log_str += ', ' + ', '.join([f'验证 {k}: {v:.6f}' for k, v in metrics.items()])

                print(exp_name)
                print(log_str)

                if writer is not None:
                    writer.add_scalar('train/loss', loss_sum / test_iter, iter_count)
                    for key, value in metrics.items():
                        writer.add_scalar(f'eval/{key}', value, iter_count)

                if 'recall' in metrics:
                    recall = metrics['recall']
                    if recall > best_metric:
                        best_metric = recall
                        model.save_model(best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        if trials > patience:
                            print(f'早停: 连续{patience}次迭代没有提升')
                            break

                loss_sum = 0.0
                test_time = time.time()
                print(f"时间间隔: {(test_time - start_time) / 60.0:.4f} 分钟")

            if iter_count >= max_iter * 1000:
                break

    except KeyboardInterrupt:
        print('提前退出训练')

    # 加载最佳模型进行评估
    model.load_model(best_model_path)

    # 验证集评估
    metrics = evaluate_full(model, valid_data, best_model_path, batch_size, item_cate_map, save=False)
    print(', '.join([f'验证 {k}: {v:.6f}' for k, v in metrics.items()]))

    # 测试集评估
    test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
    metrics = evaluate_full(model, test_data, best_model_path, batch_size, item_cate_map, save=False)
    print(', '.join([f'测试 {k}: {v:.6f}' for k, v in metrics.items()]))

    if writer is not None:        writer.close()


def test_model(
        test_file,
        cate_file,
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        lr=0.001
):
    """测试模型"""
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'

    model = get_model(dataset, model_type, item_count, batch_size, maxlen)
    if model is None:
        return

    model.load_model(best_model_path)

    item_cate_map = load_item_cate(cate_file)
    test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
    metrics = evaluate_full(model, test_data, best_model_path, batch_size, item_cate_map,
                            save=False, coef=args.coef)
    print(', '.join([f'测试 {k}: {v:.6f}' for k, v in metrics.items()]))


def output_embeddings(
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        lr=0.001
):
    """输出嵌入向量"""
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'

    model = get_model(dataset, model_type, item_count, batch_size, maxlen)
    if model is None:
        return

    model.load_model(best_model_path)
    item_embs = model.output_item()

    os.makedirs('output', exist_ok=True)
    np.save(f'output/{exp_name}_emb.npy', item_embs)
    print(f'嵌入向量已保存到 output/{exp_name}_emb.npy')


if __name__ == '__main__':
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, default='train', help='train | test | output')
    parser.add_argument('--dataset', type=str, default='book', help='book | taobao')
    parser.add_argument('--random_seed', type=int, default=19)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_interest', type=int, default=4)
    parser.add_argument('--model_type', type=str, default='DNN')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--coef', type=float, default=None)
    parser.add_argument('--topN', type=int, default=50)

    args = parser.parse_args()
    SEED = args.random_seed

    # 设置随机种子
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # 数据配置
    if args.dataset == 'taobao':
        path = r'F:\科研2\data\taobao_data'
        item_count = 1708531
        batch_size = 256
        maxlen = 50
        test_iter = 500
    elif args.dataset == 'book':
        path = r'F:\科研2\data\book_date'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    else:
        print(f"不支持的数据集: {args.dataset}")
        sys.exit(1)

    # 构建文件路径
    train_file = os.path.join(path, f'{args.dataset}_train.txt')
    valid_file = os.path.join(path, f'{args.dataset}_valid.txt')
    test_file = os.path.join(path, f'{args.dataset}_test.txt')
    cate_file = os.path.join(path, f'{args.dataset}_item_cate.txt')

    # 检查文件是否存在
    for file_path in [train_file, valid_file, test_file, cate_file]:
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在")

    # 执行相应操作
    if args.p == 'train':
        train_model(
            train_file, valid_file, test_file, cate_file, item_count, args.dataset,
            batch_size, maxlen, test_iter, args.model_type, args.learning_rate,
            args.max_iter, args.patience
        )
    elif args.p == 'test':
        test_model(
            test_file, cate_file, item_count, args.dataset, batch_size, maxlen,
            args.model_type, args.learning_rate
        )
    elif args.p == 'output':
        output_embeddings(
            item_count, args.dataset, batch_size, maxlen, args.model_type, args.learning_rate
        )
    else:
        print('未指定操作...')