# Controllable Multi-Interest Framework for Recommendation

基于多兴趣学习的推荐系统框架，实现了主流的序列推荐模型（含多兴趣模型），支持用户行为序列建模、多兴趣提取与个性化推荐，参考论文《Controllable Multi-Interest Framework for Recommendation》（KDD 2020）。czx改编

## 项目简介

本项目提供了一套完整的序列推荐模型实现方案，核心聚焦于多兴趣推荐场景——通过建模用户历史行为序列，提取用户的多个潜在兴趣偏好，解决传统推荐模型中"用户兴趣单一化"的问题，提升推荐多样性与准确性。

项目基于 TensorFlow/Keras 构建，代码结构清晰、可扩展性强，支持模型训练、权重保存/加载、嵌入向量输出等功能，可直接用于推荐系统的原型开发与实验验证。

## 数据集

数据集下载链接：[Dropbox](https://www.dropbox.com/s/m41kahhhx0a5z0u/data.tar.gz?dl=1)

下载后解压数据文件：
```bash
wget https://www.dropbox.com/s/m41kahhhx0a5z0u/data.tar.gz?dl=1 -O data.tar.gz
tar -xzf data.tar.gz
