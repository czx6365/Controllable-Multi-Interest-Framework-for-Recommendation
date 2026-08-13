import argparse
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    import faiss
except ImportError:
    faiss = None

from data_iterator import DataIterator
from model import Model_ComiRec_DR, Model_ComiRec_SA, Model_DNN, Model_GRU4REC, Model_MIND


best_metric = 0.0

# Dataset-specific training defaults belong in code; filesystem locations do not.
DATASET_DEFAULTS = {
    "book": {
        "batch_size": 128,
        "maxlen": 20,
        "test_iter": 1000,
    },
    "taobao": {
        "batch_size": 256,
        "maxlen": 50,
        "test_iter": 500,
    },
}


def prepare_data(src, target):
    """将 DataIterator 的输出整理为训练/评估需要的四元组。"""
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask


def load_item_cate(source):
    """读取 item -> category 映射。"""
    item_cate = {}
    with open(source, "r", encoding="utf-8") as file:
        for line in file:
            conts = line.strip().split(",")
            if len(conts) >= 2:
                item_cate[int(conts[0])] = int(conts[1])
    return item_cate


def infer_item_count(cate_file):
    """从类别映射文件推断 embedding table 大小，避免机器相关的硬编码 item_count。"""
    item_cate = load_item_cate(cate_file)
    if not item_cate:
        raise ValueError(f"无法从类别文件推断 item_count: {cate_file}")
    return max(item_cate) + 1


def compute_diversity(item_list, item_cate_map):
    """计算推荐列表中不同类别物品对的比例。"""
    n = len(item_list)
    if n <= 1:
        return 0.0

    diversity = 0.0
    valid_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if item_list[i] in item_cate_map and item_list[j] in item_cate_map:
                valid_pairs += 1
                diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]

    return diversity / valid_pairs if valid_pairs else 0.0


def create_faiss_index(item_embs, embedding_dim):
    """创建 FAISS 内积索引；没有 FAISS 时回退到 NumPy。"""
    if faiss is None:
        return None

    try:
        if tf.config.list_physical_devices("GPU") and hasattr(faiss, "StandardGpuResources"):
            resources = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(resources, embedding_dim, config)
        else:
            index = faiss.IndexFlatIP(embedding_dim)
        index.add(item_embs.astype(np.float32))
        return index
    except Exception as exc:
        print(f"FAISS 索引创建失败，将使用 NumPy 检索: {exc}")
        return None


def numpy_search(user_embs, item_embs, top_n):
    """FAISS 不可用时的 NumPy 内积检索。"""
    similarities = np.dot(user_embs, item_embs.T)
    indices = np.argsort(similarities, axis=1)[:, ::-1][:, :top_n]
    distances = np.take_along_axis(similarities, indices, axis=1)
    return distances, indices


def _search(index, user_embs, item_embs, top_n):
    if index is not None:
        return index.search(user_embs.astype(np.float32), top_n)
    return numpy_search(user_embs, item_embs, top_n)


def evaluate_full(model, test_data, item_cate_map, top_n, embedding_dim, save=True, coef=None):
    """计算 Recall、NDCG、HitRate，以及可选的推荐多样性。"""
    item_embs = model.output_item()
    search_index = create_faiss_index(item_embs, embedding_dim)

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0

    for src, tgt in test_data:
        _, item_id, hist_item, hist_mask = prepare_data(src, tgt)
        user_embs = model.output_user([hist_item, hist_mask])
        distances, indices = _search(search_index, user_embs, item_embs, top_n)

        if len(user_embs.shape) == 2:
            for i, iid_list in enumerate(item_id):
                true_items = set(iid_list)
                hits = 0
                dcg = 0.0
                for rank, iid in enumerate(indices[i]):
                    if iid in true_items:
                        hits += 1
                        dcg += 1.0 / math.log(rank + 2, 2)

                idcg = sum(1.0 / math.log(rank + 2, 2) for rank in range(hits))
                total_recall += hits / max(len(iid_list), 1)
                if hits:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(indices[i], item_cate_map)
        else:
            num_interest = user_embs.shape[1]
            user_embs_flat = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            distances, indices = _search(search_index, user_embs_flat, item_embs, top_n)

            for i, iid_list in enumerate(item_id):
                candidate_items = list(
                    zip(
                        np.reshape(indices[i * num_interest:(i + 1) * num_interest], -1),
                        np.reshape(distances[i * num_interest:(i + 1) * num_interest], -1),
                    )
                )
                candidate_items.sort(key=lambda pair: pair[1], reverse=True)

                ranked_items = []
                seen = set()
                if coef is None:
                    for item, _score in candidate_items:
                        if item != 0 and item not in seen:
                            ranked_items.append(item)
                            seen.add(item)
                            if len(ranked_items) >= top_n:
                                break
                else:
                    rerank_pool = []
                    for item, score in candidate_items:
                        if item not in seen and item in item_cate_map:
                            rerank_pool.append((item, score, item_cate_map[item]))
                            seen.add(item)

                    category_counts = defaultdict(int)
                    for _ in range(top_n):
                        if not rerank_pool:
                            break
                        best_index = max(
                            range(len(rerank_pool)),
                            key=lambda idx: rerank_pool[idx][1]
                            - coef * category_counts[rerank_pool[idx][2]],
                        )
                        item, _score, category = rerank_pool.pop(best_index)
                        ranked_items.append(item)
                        category_counts[category] += 1

                true_items = set(iid_list)
                hits = 0
                dcg = 0.0
                for rank, iid in enumerate(ranked_items):
                    if iid in true_items:
                        hits += 1
                        dcg += 1.0 / math.log(rank + 2, 2)

                idcg = sum(1.0 / math.log(rank + 2, 2) for rank in range(hits))
                total_recall += hits / max(len(iid_list), 1)
                if hits:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(ranked_items, item_cate_map)

        total += len(item_id)

    if total == 0:
        raise ValueError("评估数据为空。")

    metrics = {
        "recall": total_recall / total,
        "ndcg": total_ndcg / total,
        "hitrate": total_hitrate / total,
    }
    if not save:
        metrics["diversity"] = total_diversity / total
    return metrics


def get_model(dataset, model_type, item_count, batch_size, maxlen, args):
    """根据模型类型构建推荐模型。"""
    model_classes = {
        "DNN": Model_DNN,
        "GRU4REC": Model_GRU4REC,
        "MIND": Model_MIND,
        "ComiRec-DR": Model_ComiRec_DR,
        "ComiRec-SA": Model_ComiRec_SA,
    }
    if model_type not in model_classes:
        raise ValueError(f"不支持的模型类型: {model_type}")

    if model_type == "MIND":
        model = model_classes[model_type](
            item_count,
            args.embedding_dim,
            args.hidden_size,
            batch_size,
            args.num_interest,
            maxlen,
            relu_layer=(dataset == "book"),
        )
    elif model_type in {"ComiRec-DR", "ComiRec-SA"}:
        model = model_classes[model_type](
            item_count,
            args.embedding_dim,
            args.hidden_size,
            batch_size,
            args.num_interest,
            maxlen,
        )
    else:
        model = model_classes[model_type](
            item_count,
            args.embedding_dim,
            args.hidden_size,
            batch_size,
            maxlen,
        )

    # Keras 子类模型会在第一次前向传播时真正创建权重。
    dummy_history = np.zeros((1, maxlen), dtype=np.int32)
    dummy_mask = np.ones((1, maxlen), dtype=np.float32)
    model([dummy_history, dummy_mask], training=False)
    return model


def get_exp_name(dataset, model_type, batch_size, lr, maxlen, experiment_name):
    """生成非交互式实验名称，便于脚本化和复现。"""
    base = "_".join(
        [dataset, model_type, f"b{batch_size}", f"lr{lr}", f"d{args.embedding_dim}", f"len{maxlen}"]
    )
    return f"{base}_{experiment_name}" if experiment_name else base


def setup_gpu():
    """按需启用 TensorFlow GPU memory growth。"""
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)


def train_model(
    train_file,
    valid_file,
    test_file,
    cate_file,
    item_count,
    dataset,
    batch_size,
    maxlen,
    test_iter,
    model_type,
    lr,
    max_iter,
    patience,
    args,
):
    """训练并在验证集上早停，最后报告验证/测试指标。"""
    global best_metric
    best_metric = 0.0

    exp_name = get_exp_name(
        dataset, model_type, batch_size, lr, maxlen, args.experiment_name
    )
    best_model_path = os.path.join("best_model", exp_name)

    if args.overwrite and os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)

    setup_gpu()
    item_cate_map = load_item_cate(cate_file)
    train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
    valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1)

    model = get_model(dataset, model_type, item_count, batch_size, maxlen, args)

    print(f"开始训练: {exp_name}")
    start_time = time.time()
    iter_count = 0
    loss_sum = 0.0
    trials = 0

    try:
        for src, tgt in train_data:
            nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)
            loss_dict = model.train_step([nick_id, item_id, hist_item, hist_mask, lr])
            loss_sum += float(loss_dict["loss"])
            iter_count += 1

            if iter_count % test_iter == 0:
                metrics = evaluate_full(
                    model,
                    valid_data,
                    item_cate_map,
                    args.topN,
                    args.embedding_dim,
                    save=True,
                )
                print(
                    f"iter={iter_count} loss={loss_sum / test_iter:.4f} "
                    + " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                )

                recall = metrics["recall"]
                if recall > best_metric:
                    best_metric = recall
                    model.save_model(best_model_path)
                    trials = 0
                else:
                    trials += 1
                    if trials > patience:
                        print(f"早停: 连续 {patience} 次评估没有提升")
                        break

                loss_sum = 0.0
                print(f"elapsed_min={(time.time() - start_time) / 60.0:.2f}")

            if iter_count >= max_iter * 1000:
                break
    except KeyboardInterrupt:
        print("训练被用户中断。")

    if not os.path.exists(best_model_path):
        model.save_model(best_model_path)

    model.load_model(best_model_path)
    valid_metrics = evaluate_full(
        model, valid_data, item_cate_map, args.topN, args.embedding_dim, save=False
    )
    test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
    test_metrics = evaluate_full(
        model, test_data, item_cate_map, args.topN, args.embedding_dim, save=False
    )

    print("验证:", ", ".join(f"{k}={v:.6f}" for k, v in valid_metrics.items()))
    print("测试:", ", ".join(f"{k}={v:.6f}" for k, v in test_metrics.items()))


def test_model(test_file, cate_file, item_count, dataset, batch_size, maxlen, model_type, args):
    """加载已有权重并评估。"""
    exp_name = get_exp_name(
        dataset, model_type, batch_size, args.learning_rate, maxlen, args.experiment_name
    )
    best_model_path = os.path.join("best_model", exp_name)
    model = get_model(dataset, model_type, item_count, batch_size, maxlen, args)
    model.load_model(best_model_path)

    item_cate_map = load_item_cate(cate_file)
    test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
    metrics = evaluate_full(
        model,
        test_data,
        item_cate_map,
        args.topN,
        args.embedding_dim,
        save=False,
        coef=args.coef,
    )
    print(", ".join(f"{k}={v:.6f}" for k, v in metrics.items()))


def output_embeddings(item_count, dataset, batch_size, maxlen, model_type, args):
    """导出训练后的 item embeddings。"""
    exp_name = get_exp_name(
        dataset, model_type, batch_size, args.learning_rate, maxlen, args.experiment_name
    )
    best_model_path = os.path.join("best_model", exp_name)
    model = get_model(dataset, model_type, item_count, batch_size, maxlen, args)
    model.load_model(best_model_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{exp_name}_emb.npy"
    np.save(output_path, model.output_item())
    print(f"嵌入向量已保存到: {output_path}")


def resolve_data_config(args):
    """解析可移植的数据路径与数据集运行参数。"""
    defaults = DATASET_DEFAULTS[args.dataset]
    data_dir = Path(
        args.data_dir
        or os.getenv("REC_DATA_DIR", "")
        or Path("data") / args.dataset
    ).expanduser()

    files = {
        "train": data_dir / f"{args.dataset}_train.txt",
        "valid": data_dir / f"{args.dataset}_valid.txt",
        "test": data_dir / f"{args.dataset}_test.txt",
        "cate": data_dir / f"{args.dataset}_item_cate.txt",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        expected = "\n  - ".join(missing)
        raise FileNotFoundError(
            "推荐数据文件不完整。通过 --data-dir 或 REC_DATA_DIR 指定数据目录。"
            f"\n缺少:\n  - {expected}"
        )

    item_count = args.item_count or infer_item_count(files["cate"])
    batch_size = args.batch_size or defaults["batch_size"]
    maxlen = args.maxlen or defaults["maxlen"]
    test_iter = args.test_iter or defaults["test_iter"]
    return files, item_count, batch_size, maxlen, test_iter


def build_parser():
    parser = argparse.ArgumentParser(
        description="Portable training/evaluation entry point for the multi-interest recommendation study."
    )
    parser.add_argument("-p", choices=["train", "test", "output"], default="train")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="book")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset directory. Falls back to REC_DATA_DIR, then data/<dataset>.",
    )
    parser.add_argument("--item-count", type=int, default=None, help="Override inferred item count.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--maxlen", type=int, default=None)
    parser.add_argument("--test-iter", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=19)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-interest", type=int, default=4)
    parser.add_argument(
        "--model-type",
        choices=["DNN", "GRU4REC", "MIND", "ComiRec-DR", "ComiRec-SA"],
        default="DNN",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--coef", type=float, default=None)
    parser.add_argument("--topN", type=int, default=50)
    parser.add_argument("--experiment-name", type=str, default="run")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    try:
        files, item_count, batch_size, maxlen, test_iter = resolve_data_config(args)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    if args.p == "train":
        train_model(
            str(files["train"]),
            str(files["valid"]),
            str(files["test"]),
            str(files["cate"]),
            item_count,
            args.dataset,
            batch_size,
            maxlen,
            test_iter,
            args.model_type,
            args.learning_rate,
            args.max_iter,
            args.patience,
            args,
        )
    elif args.p == "test":
        test_model(
            str(files["test"]),
            str(files["cate"]),
            item_count,
            args.dataset,
            batch_size,
            maxlen,
            args.model_type,
            args,
        )
    else:
        output_embeddings(
            item_count,
            args.dataset,
            batch_size,
            maxlen,
            args.model_type,
            args,
        )
