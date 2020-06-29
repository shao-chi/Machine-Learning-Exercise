# 1082 NCTU Deep Learning Competition
### 新聞標題文類

### Data Preprocess
* 利用`jieba`將標題與關鍵字斷詞 
* 依分類建立字典，並選取10000最常出現的字，娛樂類別選20000個字
* 將標題與關鍵字串接
* 利用`gensim.models.word2vec`將斷好的詞轉成向量

### Model
* ~~CNN~~
* ~~LSTM~~
* **Fully-Connect NN**

### 執行方式
* Training model <br>
```python3 train.py```