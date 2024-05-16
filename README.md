# Empirical Evaluation of Variational Autoencoders and Denoising Diffusion Models for Data Augmentation in Bioacoustics Classification

>**Abstract:** One major challenge in supervised deep learning is the need for large training datasets to achieve satisfactory generalisation performance. Acquiring audio recordings of endangered animals compounds this issue due to high costs, logistical constraints, and the rarity of the species in question. Typically, bioacoustic datasets have imbalanced class distributions, further complicating model training with limited examples for some rare species. To overcome this, our study proposes the evaluation of generative models for audio augmentation. Generative models, such as Variational Autoencoders (VAEs) and Denoising Diffusion Probabilistic Models (DDPMs), offer the ability to create synthetic data after training on existing datasets. We assess the effectiveness of VAEs and DDPMs in augmenting a bioacoustics dataset, which includes vocalisations of the world's rarest primate, the Hainan gibbon. We assess the generated synthetic data through visual inspection and by computing the Kernel Inception Distance, to compare the distribution of the generated dataset to the training set. Furthermore, we investigate the efficacy using the generated dataset to train a deep learning classifier to identify the Hainan gibbon calls. We vary the size of the training datasets and compare the classification performance across four scenarios: no augmentation, augmentation with VAEs, augmentation with DDPMs, and standard bioacoustics augmentation methods. Our study is the first to show that standard audio augmentation methods are as effective as newer generative approaches commonly used in computer vision. Considering the high computational costs of VAEs and DDPMs, this emphasizes the suitability of simpler techniques for building deep learning classifiers on bioacoustic datasets.
----
## Data

----
## Requirements
Set up and activate the virtual [conda](https://docs.anaconda.com/anaconda/install/index.html) environment: 
- `conda env create -f environment.yml`
- `conda activate gdbc`
