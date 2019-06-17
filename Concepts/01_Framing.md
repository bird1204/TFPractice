# Framing

## 什麼是監督式學習？
學習要怎麼組合前所未見的資料，以產生有用的預測


## Labels:
label 是我們要預測的東西，就是在線性迴歸裡的 y 變數

## Features:
feature 是輸入的變數，就是在線性迴歸裡的 x 變數，越複雜的 ML 可以用到更多的 features

## Examples:
example 是資料的實例( 用 **x** 表示向量 )，並且細分成 :  

1. labeled examples : 有 y 的   
2. unlabeled examples : 沒有 y 的

## Models:
model 定義了 feature / label 之間的關係，model 的生命週期又分成：  

1. Training 的意思是建立一個 model，讓 model 知道 feature / label 之間的關係  
2. Inference 的意思是套用訓練好的 model 在 unlabeled examples，用來預測 y

## Regression vs. classification:
問題有分成兩種：  

1. regression: 迴歸問題，預測具有連續值的問題  
	( 有多少機率？ / 有多少價值？ )  
2. classification: 分類問題，預測離散值的問題  
	( 有沒有？ / 會不會？ / 是不是？ )