## Regularization of deep multi-task networks using orthogonal gradients
This repo contains a short Keras implementation of the following [paper](https://arxiv.org/abs/1912.06844):

```
Suteu, M.; Guo Y. Regularizing deep multi-task networks using orthogonal gradients. arXiv preprint, arXiv:1912.06844, 2019. 
```

Disclaimer: This is my own interpretation of what the authors described in text. It may not be entirely faithful to their own implementation, particularly since some network hyperparameters are not provided by the authors. 


Note: This script requires the MultiDigitMNIST dataset:
```
Sun, S. Multi-digit MNIST for Few-shot Learning, github repo, 2019. 
```
Once you have generated this dataset, you may run the assemble_dataset.py script to group data in a manner similar to what was done in the above paper. The MultiDigitMNIST dataset is available [here](https://github.com/shaohua0116/MultiDigitMNIST).








