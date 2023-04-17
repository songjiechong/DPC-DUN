# Dynamic Path-Controllable Deep Unfolding Network for Compressive Sensing (IEEE TIP 2023)
This repository is for DPC-DUN introduced in the following paper：

[Jiechong Song](https://scholar.google.com/citations?hl=en&user=EBOtupAAAAAJ), Bin Chen and [Jian Zhang](http://jianzhang.tech/), "Dynamic Path-Controllable Deep Unfolding Network for Compressive Sensing ", in the IEEE Transactions on Image Processing (TIP), 2023. [PDF](https://ieeexplore.ieee.org/document/10098557)

## :art: Abstract
Deep unfolding network (DUN) that unfolds the optimization algorithm into a deep neural network has achieved great success in compressive sensing (CS) due to its good interpretability and high performance. Each stage in DUN corresponds to one iteration in optimization. At the test time, all the sampling images generally need to be processed by all stages, which comes at a price of computation burden and is also unnecessary for the images whose contents are easier to restore. In this paper, we focus on CS reconstruction and propose a novel Dynamic Path-Controllable Deep Unfolding Network (DPC-DUN). DPC-DUN with our designed path-controllable
selector can dynamically select a rapid and appropriate route for each image and is slimmable by regulating different performance-complexity tradeoffs. Extensive experiments show that our DPC-DUN is highly flexible and can provide excellent performance and dynamic adjustment to get a suitable tradeoff, thus addressing the main requirements to become appealing in practice.

## :fire: Network Architecture
![Network](/Figs/network.png)

## 🚩 Results
###  Qualitative Evaluation
![Set11](/Figs/set11.png)
![CBSD_DIV2K](/Figs/Urban_DIV2K.png)
### The controllable effect of DPC-DUN
![DPC-DUN](/Figs/controllable.png)

## 🔧 Requirements
- Python == 3.8.5
- Pytorch == 1.8.0

## 👀 Datasets
- Train data: [train400](https://drive.google.com/file/d/15FatS3wYupcoJq44jxwkm6Kdr0rATPd0/view?usp=sharing)
- Test data: Set11, [CBSD68](https://drive.google.com/file/d/1Q_tcV0d8bPU5g0lNhVSZXLFw0whFl8Nt/view?usp=sharing), [Urban100](https://drive.google.com/file/d/1cmYjEJlR2S6cqrPq8oQm3tF9lO2sU0gV/view?usp=sharing), [DIV2K](https://drive.google.com/file/d/1olYhGPuX8QJlewu9riPbiHQ7XiFx98ac/view?usp=sharing)

## 📑 Citation
If you find our work helpful in your resarch or work, please cite the following paper.
```
@article{song2023dynamic,
  title={Dynamic Path-Controllable Deep Unfolding Network for Compressive Sensing},
  author={Song, Jiechong and Chen, Bin and Zhang, Jian},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```

## :e-mail: Contact
If you have any question, please email `songjiechong@pku.edu.cn`.

## :hugs: Acknowledgements
This code is built on [ISTA-Net-PyTorch](https://github.com/jianzhangcs/ISTA-Net-PyTorch). We thank the authors for sharing their codes.




