# UIS_Trade-off
This repository is the official implementation of the paper:
* On the Utility-Informativeness-Security Trade-off in Discrete Task-Oriented Semantic Communication
* Authors: Anbang Zhang, Yanhu Wang, and Shuaishuai Guo
  
## Design Philosophy
### This Work: Synergistic Alignment of Learning and Communication Objectives
![9e610f5be3085d293deddebdf8fa618](https://github.com/zab0613/UIS_Trade-off/assets/117052094/9c944c6c-5996-45c5-9223-062cfc1173bd)
In detail, most existing frameworks for task-oriented semantic communications predominantly concentrate on individual aspects such as edge inference performance, transmission enhancement, or security improvement. Rarely do these frameworks address all three elements concurrently. Our UIS-ToSC framework stands out by combining edge inference utility, informativeness, and enhanced security. By incorporating the information bottleneck principle, vector quantization loss, MSE loss, and perceptual loss, each focusing on different metrics, one can find a more nuanced balance between these competing objectives and tailor the training process to domain-specific needs more effectively. Moreover, we introduce adversarial learning into UIS-ToSC ensuring that the designed codebook possesses intrinsic security properties, setting our work apart from conventional methods.

## Training
Firstly, train the UIS-ToSC framework.
```
python /run_VQVAE.py
----main_train()
```
Then, train the attacker stealing models.
```
python /run_VQVAE.py
----main_train_dec()
```

## Simulation dataest
CIFAR-10 dataest

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
