import os
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')


# 使用 tf.keras 而不是直接导入
class BaseModel(tf.keras.Model):
    """基础模型类，其他推荐模型的父类"""

    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
        super(BaseModel, self).__init__()  # 调用父类（Keras.Model）的初始化方法
        self.model_flag = flag  # 保存模型类型标识
        self.reg = False  # 是否启用正则化（默认关闭，后续可扩展）
        self.batch_size = batch_size  # 保存批次大小
        self.n_mid = n_mid  # 保存物品总数
        self.embedding_dim = embedding_dim  # 保存嵌入向量维度
        self.hidden_size = hidden_size  # 保存隐藏层神经元数量
        self.seq_len = seq_len  # 保存序列长度
        self.neg_num = 10  # 负样本数量（推荐系统中常用，用于对比学习）

        # 嵌入层
        self.mid_embeddings = tf.keras.layers.Embedding(#创建嵌入层，将离散的整数输入转换为密集的向量表示
            n_mid,#input_dim参数，表示词汇表的大小或输入整数的范围
            embedding_dim,#output_dim参数，被映射到的密集向量的维度
            embeddings_initializer=tf.random_normal_initializer(stddev=0.1),
            name="mid_embedding"
        )
        self.mid_embeddings_bias = tf.Variable(
            tf.zeros([n_mid]),
            name="bias_lookup_table",
            trainable=False
        )

    def call(self, inputs, training=None):
        """前向传播"""
        raise NotImplementedError("子类必须实现call方法")

    def build_sampled_softmax_loss(self, user_emb, item_ids):
        """构建采样的softmax损失函数"""
        # 使用标准的交叉熵损失
        logits = tf.matmul(user_emb, self.mid_embeddings.weights[0], transpose_b=True)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=item_ids,
                logits=logits
            )
        )
        return loss

    def train_step(self, data):#模型训练的一个单步过程，包括数据处理、前向传播、损失计算、反向传播和参数更新，最后返回训练损失值
        """训练步骤"""
        # 确保数据格式正确
        if len(data) == 5:
            uid_batch, mid_batch, mid_his_batch, mask_batch, lr = data
        else:
            # 如果数据格式不对，尝试重新组织
            if len(data) == 4:
                uid_batch, mid_batch, mid_his_batch, mask_batch = data
                lr = 0.001
            else:
                # 如果只有3个参数，假设是 mid_batch, mid_his_batch, mask_batch
                mid_batch, mid_his_batch, mask_batch = data
                uid_batch = tf.zeros_like(mid_batch)  # 创建虚拟的uid
                lr = 0.001

        # 确保所有输入都是张量
        mid_his_batch = tf.convert_to_tensor(mid_his_batch, dtype=tf.int32)
        mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)
        mid_batch = tf.convert_to_tensor(mid_batch, dtype=tf.int32)

        with tf.GradientTape() as tape:
            # 前向传播 - 确保正确传递输入
            user_emb = self.call([mid_his_batch, mask_batch], training=True)

            # 计算损失
            loss = self.build_sampled_softmax_loss(user_emb, mid_batch)

        # 应用梯度
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'loss': loss}

    def output_item(self):
        """输出物品嵌入向量"""
        return self.mid_embeddings.weights[0].numpy()

    def output_user(self, inputs):
        """输出用户嵌入向量"""
        mid_his_batch, mask_batch = inputs

        # 确保输入是张量
        mid_his_batch = tf.convert_to_tensor(mid_his_batch, dtype=tf.int32)
        mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)

        user_emb = self.call([mid_his_batch, mask_batch], training=False)
        return user_emb.numpy()

    def save_model(self, path):
        """保存模型"""
        if not os.path.exists(path):
            os.makedirs(path)
        # 使用正确的文件扩展名
        weights_path = os.path.join(path, 'model_weights.weights.h5')
        self.save_weights(weights_path)
        print(f'模型权重已保存到: {weights_path}')

    def load_model(self, path):
        """加载模型"""
        weights_path = os.path.join(path, 'model_weights.weights.h5')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)
            print(f'模型已从 {weights_path} 恢复')
        else:
            print(f'警告: 未找到模型权重文件 {weights_path}')


class Model_DNN(BaseModel):
    """基于DNN deep neural network的推荐模型"""

    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_DNN, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN")

        # DNN层    全连接层
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu', name='dense1')
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation='relu', name='dense2')

        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, inputs, training=None):  #call是模型前向传播的强制接口
        # 解包输入，inputs是包含“历史物品序列”和“掩码”的列表
        mid_his_batch, mask_batch = inputs
        #mid_his_batch：用户历史物品 ID 序列，形状为[batch_size, seq_len]（例：[32, 256]表示 “32 个用户，每个用户的历史行为序列长度为 256”）；
        #mask_batch：掩码向量，[batch_size, seq_len]，用于处理 “序列长度不足 seq_len” 的情况（如部分用户历史行为只有 10 个物品，其余 246 个位置用 0 填充，掩码对应位置为 1 表示 “有效行为”，0 表示 “填充无效”）

        # 确保输入是张量
        mid_his_batch = tf.convert_to_tensor(mid_his_batch, dtype=tf.int32)
        mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)

        # 获取嵌入
        item_his_eb = self.mid_embeddings(mid_his_batch)  # 输出形状：(batch_size, seq_len, embedding_dim)

        # 应用mask
        mask = tf.expand_dims(mask_batch, -1)  # (batch_size, seq_len, 1)
        item_his_eb = item_his_eb * mask

        # 平均池化
        sum_emb = tf.reduce_sum(item_his_eb, 1)  # (batch_size, embedding_dim)
        sum_mask = tf.reduce_sum(mask_batch, 1, keepdims=True)  # (batch_size, 1)
        user_emb = sum_emb / (sum_mask + 1e-8)

        # DNN层
        user_emb = self.dense1(user_emb)
        user_emb = self.dense2(user_emb)

        return user_emb


class Model_GRU4REC(BaseModel):
    """基于GRU的推荐模型"""

    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_GRU4REC, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="GRU4REC")

        # GRU层
        self.gru = tf.keras.layers.GRU(
            hidden_size,
            return_sequences=True,
            return_state=True,
            name='gru_layer'
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, inputs, training=None):
        # 解包输入
        mid_his_batch, mask_batch = inputs

        # 确保输入是张量
        mid_his_batch = tf.convert_to_tensor(mid_his_batch, dtype=tf.int32)
        mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)

        # 获取嵌入
        item_his_eb = self.mid_embeddings(mid_his_batch)  # (batch_size, seq_len, embedding_dim)

        # 应用mask
        mask = tf.expand_dims(mask_batch, -1)
        item_his_eb = item_his_eb * mask

        # 序列长度
        sequence_length = tf.cast(tf.reduce_sum(mask_batch, -1), dtype=tf.int32)

        # GRU处理
        rnn_outputs, final_state = self.gru(
            item_his_eb,
            mask=tf.sequence_mask(sequence_length, self.seq_len)
        )

        return final_state


class CapsuleNetwork(tf.keras.layers.Layer):
    """胶囊网络层"""

    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type#双线性变换类型(0,1,2)
        self.num_interest = num_interest#兴趣胶囊数量
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def build(self, input_shape):
        # 构建权重
        if self.bilinear_type == 0:
            self.dense_0 = tf.keras.layers.Dense(self.dim, activation=None, use_bias=False)
        elif self.bilinear_type == 1:
            self.dense_1 = tf.keras.layers.Dense(self.dim * self.num_interest, activation=None, use_bias=False)
        else:
            self.w = self.add_weight(
                name='weights',
                shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                initializer=tf.random_normal_initializer(stddev=0.1)
            )

        if self.relu_layer:
            self.proj_layer = tf.keras.layers.Dense(self.dim, activation=tf.nn.relu, name='proj')

        super(CapsuleNetwork, self).build(input_shape)

    def call(self, inputs):
        item_his_emb, item_eb, mask = inputs

        batch_size = tf.shape(item_his_emb)[0]

        with tf.name_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = self.dense_0(item_his_emb)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = self.dense_1(item_his_emb)
            else:
                u = tf.expand_dims(item_his_emb, axis=2)
                item_emb_hat = tf.reduce_sum(self.w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [batch_size, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [batch_size, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([batch_size, self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(
                tf.random.normal([batch_size, self.num_interest, self.seq_len], stddev=1.0)
            )

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [batch_size, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [batch_size, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = self.proj_layer(interest_capsule)

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [batch_size, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [batch_size, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(
                tf.reshape(interest_capsule, [batch_size * self.num_interest, self.dim]),
                tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(batch_size) * self.num_interest
            )
        else:
            readout = tf.matmul(tf.reshape(atten, [batch_size, 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [batch_size, self.dim])

        return interest_capsule, readout


class Model_MIND(BaseModel):
    """MIND模型：多兴趣网络"""

    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, hard_readout=True,
                 relu_layer=True):
        super(Model_MIND, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="MIND")

        self.num_interest = num_interest
        self.capsule_network = CapsuleNetwork(
            hidden_size, seq_len,
            bilinear_type=0,
            num_interest=num_interest,
            hard_readout=hard_readout,
            relu_layer=relu_layer
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, inputs, training=None):
        # 解包输入
        mid_his_batch, mask_batch = inputs

        # 确保输入是张量
        mid_his_batch = tf.convert_to_tensor(mid_his_batch, dtype=tf.int32)
        mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)

        # 获取嵌入
        item_his_emb = self.mid_embeddings(mid_his_batch)

        # 创建虚拟物品嵌入用于注意力计算
        batch_size = tf.shape(mid_his_batch)[0]
        dummy_items = tf.zeros([batch_size], dtype=tf.int32)
        item_emb = self.mid_embeddings(dummy_items)

        # 胶囊网络
        user_eb, readout = self.capsule_network([item_his_emb, item_emb, mask_batch])

        return readout


class Model_ComiRec_DR(BaseModel):
    """ComiRec-DR模型"""

    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, hard_readout=True,
                 relu_layer=False):
        super(Model_ComiRec_DR, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len,
                                               flag="ComiRec_DR")

        self.num_interest = num_interest
        self.capsule_network = CapsuleNetwork(
            hidden_size, seq_len,
            bilinear_type=2,
            num_interest=num_interest,
            hard_readout=hard_readout,
            relu_layer=relu_layer
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, inputs, training=None):
        # 解包输入
        mid_his_batch, mask_batch = inputs

        # 确保输入是张量
        mid_his_batch = tf.convert_to_tensor(mid_his_batch, dtype=tf.int32)
        mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)

        # 获取嵌入
        item_his_emb = self.mid_embeddings(mid_his_batch)

        # 创建虚拟物品嵌入用于注意力计算
        batch_size = tf.shape(mid_his_batch)[0]
        dummy_items = tf.zeros([batch_size], dtype=tf.int32)
        item_emb = self.mid_embeddings(dummy_items)

        # 胶囊网络
        user_eb, readout = self.capsule_network([item_his_emb, item_emb, mask_batch])

        return readout


class Model_ComiRec_SA(BaseModel):
    """ComiRec-SA模型"""

    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=True):
        super(Model_ComiRec_SA, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len,
                                               flag="ComiRec_SA")

        self.num_interest = num_interest
        self.dim = embedding_dim
        self.add_pos = add_pos
        self.seq_len = seq_len

        if add_pos:
            self.position_embedding = tf.Variable(
                tf.random.normal([1, seq_len, embedding_dim], stddev=0.1),
                name='position_embedding'
            )

        self.att_w_dense1 = tf.keras.layers.Dense(hidden_size * 4, activation=tf.nn.tanh, name='att_w_dense1')
        self.att_w_dense2 = tf.keras.layers.Dense(num_interest, activation=None, name='att_w_dense2')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, inputs, training=None):
        # 解包输入
        mid_his_batch, mask_batch = inputs

        # 确保输入是张量
        mid_his_batch = tf.convert_to_tensor(mid_his_batch, dtype=tf.int32)
        mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)

        batch_size = tf.shape(mid_his_batch)[0]

        # 获取嵌入
        item_list_emb = self.mid_embeddings(mid_his_batch)  # (batch_size, seq_len, embedding_dim)

        if self.add_pos:
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [batch_size, 1, 1])
        else:
            item_list_add_pos = item_list_emb

        num_heads = self.num_interest
        with tf.name_scope("self_atten"):
            item_hidden = self.att_w_dense1(item_list_add_pos)
            item_att_w = self.att_w_dense2(item_hidden)
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(mask_batch, axis=1), [1, num_heads, 1])
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            item_att_w = tf.nn.softmax(item_att_w)

            interest_emb = tf.matmul(item_att_w, item_list_emb)

        # 注意力机制选择主要兴趣
        dummy_items = tf.zeros([batch_size], dtype=tf.int32)
        item_emb = self.mid_embeddings(dummy_items)

        atten = tf.matmul(interest_emb, tf.reshape(item_emb, [batch_size, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [batch_size, num_heads]), 1))

        readout = tf.gather(
            tf.reshape(interest_emb, [batch_size * num_heads, self.dim]),
            tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(batch_size) * num_heads
        )

        return readout


# 模型工厂函数
def get_model(model_type, n_mid, embedding_dim, hidden_size, batch_size, seq_len, num_interest=4):
    """根据模型类型创建模型实例"""
    model_classes = {
        'DNN': Model_DNN,
        'GRU4REC': Model_GRU4REC,
        'MIND': Model_MIND,
        'ComiRec-DR': Model_ComiRec_DR,
        'ComiRec-SA': Model_ComiRec_SA
    }

    if model_type not in model_classes:
        raise ValueError(f"不支持的模型类型: {model_type}")

    if model_type in ['MIND', 'ComiRec-DR', 'ComiRec-SA']:
        return model_classes[model_type](
            n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len
        )
    else:
        return model_classes[model_type](
            n_mid, embedding_dim, hidden_size, batch_size, seq_len
        )