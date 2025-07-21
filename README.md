# 1.-Neural-Collaborative-Filtering-NCF-with-Deep-Learning
A PyTorch implementation of Neural Collaborative Filtering (NCF)—a deep learning framework for learning user–item interactions via neural networks. It implements three main architectures:

GMF: Generalized Matrix Factorization

MLP: Multi-Layer Perceptron

NeuMF: Neural Matrix Factorization (fusion of GMF and MLP)

This implementation is designed for implicit-feedback scenario (e.g., clicks, views) and optimizes models with negative sampling & binary log loss.
# 2.-Background
The NCF framework was proposed by Xiangnan He et al. in “Neural Collaborative Filtering” (WWW 2017). It replaces the conventional inner product (used in matrix factorization) with a trainable neural network to better capture the complex latent interactions between users and items 
Reddit
+7
liqiangnie.github.io
+7
GitHub
+7
arXiv
+2
GitHub
+2
GitHub
+2
GitHub
+4
arXiv
+4
liqiangnie.github.io
+4
.

Key points:

Leverages neural networks to learn flexible interaction functions.

Includes three distinct architectures: GMF, MLP, and the hybrid NeuMF.

Optimizes for implicit feedback using log-loss with negative sampling.

Demonstrates improved performance (e.g., Hit Ratio, NDCG) over traditional MF 
arXiv
+2
liqiangnie.github.io
+2
Medium
+2
GitHub
GitHub
+1
GitHub
+1
.
# 3-🚀 Features
Dataset support: MovieLens 1M (configurable for other datasets)

User–item embedding via:

GMF: element-wise product

MLP: multiple dense layers

NeuMF: concatenated GMF + MLP output

Loss: binary cross‑entropy with negative sampling

Metrics: Hit Ratio (HR), Normalized Discounted Cumulative Gain (NDCG)

Hardware support: GPU (via CUDA) and CPU

Easy configuration: adjust hyperparameters directly in train.py or via CLI/config
# 4.-🔁 Workflow
The NCF workflow involves the following key steps:

1. Data Loading & Preprocessing
Load user-item interaction data (e.g., from MovieLens)

Convert into implicit feedback (positive = 1, unobserved = 0)

Perform negative sampling to generate negative user-item pairs

Split dataset into training and test sets

2. Model Selection
Choose one of the three architectures:

GMF: Element-wise product of user and item embeddings

MLP: Pass concatenated embeddings through deep neural network

NeuMF: Combine outputs from GMF and MLP

3. Training
Input: user IDs, item IDs, and labels (1 for positive, 0 for negative)

Forward pass through the selected model

Loss: Binary Cross-Entropy (BCE) between predicted and true labels

Optimizer: Adam (default), updating embedding and MLP weights

Repeat for n epochs

4. Evaluation
Use Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG)

Evaluate ranking of positive items among randomly sampled negatives

5. Inference
Recommend top-N items for each user by ranking predicted scores

Can be exported for building recommendation UI or APIs
graph TD
    A[Start] --> B[Load Dataset]
    B --> C[Preprocess & Negative Sampling]
    C --> D[Choose Model: GMF / MLP / NeuMF]
    D --> E[Train Model (BCE + Adam)]
    E --> F[Evaluate: HR / NDCG]
    F --> G[Save Model / Inference]
    G --> H[Recommend Top-N Items]
    H --> I[End]
# 5.-🔍 Insights & Tips
MLP depth: deeper architectures offer stronger representation but may need regularization 
arXiv
+1
GitHub
+1
GitHub
.

NeuMF pretraining: initialising from pretrained GMF/MLP can accelerate convergence and improve results 
liqiangnie.github.io
+2
GitHub
+2
GitHub
+2
.

Negative sampling ratio & reg: tuning these can greatly affect HR/NDCG.

Scalability: works well for moderate-size datasets; might need batching or sampling strategies for large corpora.
