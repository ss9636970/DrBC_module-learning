# LetNet (numpy)

# 1.  training dataset

此任務為圖片分類任務，資料皆為狗的照片，並可以分為50種不同的狗，下方照片為舉例圖片

![n02111277_160](https://github.com/ss9636970/KAZE-perception_learning/blob/main/readme/n02111277_160.JPEG)![n02111277_160](https://github.com/ss9636970/KAZE-perception_learning/blob/main/readme/n02111500_113.jpg)



資料可分為:

63325張訓練資料

450張測試資料

450張驗證資料



# 2. LetNet model
本篇使用Letnet為訓練模型，實作時以numpy為主要套件。

Letnet模型為多個convolution彼此串聯為特徵提取部分，後街多個全連階層為分類模型。

下突圍letNet結構圖

![LetNet](https://github.com/ss9636970/numpy-LeNet/blob/main/readme/letnet.png)



# 3. 程式碼說明

LeNet_module.py 為LeNet模型定義程式碼

funcion.py 為程式中運用到的函式。

CNN_encoder.ipynb為訓練特徵提取執行程式

main.ipynb為執行模型訓練程式，當中包括讀取資料及特模型訓練的程式碼





