# Multiscale Contrastive Learning for Node Clustering Based on Variational Graph Auto-Encoder

This repository provides an improved implementation of the Multiscale Contrastive Variational Graph Auto-Encoder (MCVGAE).

This implementation modifies the original CVGAE implementation by replacing the `compute_separtion_loss` function with an updated version.

All other components of the original implementation have been retained.

Variational graph auto-encoders (VGAEs) are a key tool for node clustering, but existing models face several significant challenges. These challenges include a mismatch between inference and generative models after incorporating the clustering inductive bias, as well as posterior collapse (PC), where latent representations become overly influenced by the prior distribution. In addition, in existing VGAEs, noisy clustering assignments lead to the feature randomness (FR) challenge, while the strong tradeoff between clustering accuracy and reconstruction quality results in the feature drift (FD) problem. To address these issues, we propose a multiscale contrastive VGAE (MCVGAE). This multiscale model combines cluster-level and graph-level contrastive learning with proximity-level and clusterlevel self supervised methods. MCVGAE improves the alignment between the hidden space and the data distribution and prevents PC. Moreover, it reduces FR and FD more effectively than existing techniques. Achieving impressive accuracy scores of 79.09% on Cora, 90.04% on ACM, 75.12% on Pubmed, 72.7% on Citeseer, 74.11% on DBLP, and 59.79% on Wiki clearly demonstrates the superiority of MCVGAE over 30 state-of-theart methods.


