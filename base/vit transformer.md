## ViT 论文

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## 介绍

在计算机视觉领域中，多数算法都是保持 CNN 整体结构不变，在 CNN 中增加 attention 模块或者使用 attention 模块替换 CNN 中的某些部分。研究者提出，没有必要总依赖于 CNN 。因此，作者提出 ViT ，仅仅使用 Transformer 结构也能够在图像分类任务中表现很好。

受到 NLP 领域中 Transformer 成功应用的启发，ViT 中尝试将标准的 Transformer 结构直接应用于图像，并对整个图像分类流程进行最少的修改。

具体来讲，**ViT 中，会将整幅图像拆分成小图像块，然后把这些小图像块的线性嵌入序列作为 Transformer 的输入送入网络**，然后使用监督学习的方式进行图像分类的训练。

## 模型结构

ViT 的整体结构如图下所示：

![vit_transforermer_p1](../assets/vit-transformer/vit-transforermer-p1.png)

## image embedding patching

在 Transformer 结构中，输入是一个二维的矩阵，矩阵的形状可以表示为 