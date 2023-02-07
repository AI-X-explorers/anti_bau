## Pre-training-based protein affinity prediction

### Step1: Preprocess
Complete the preprocessing process in utils/preprocess.py  

**Data required:**
1. action data.
2. protein sequence data.

```Python
CUDA_VISIBLE_DEVICES=1 python utils/preprocess.py
```
#### **1. Generate dataset**
In order to generate ppi structured data, you need to generate a file with utils/preprocess.py 

You will get structured dataset in folder data, which has these columns:
* **item_id_a:** Target protein A
* **item_id_b:** Protein B to interact with protein A.
* **sequence_a:** Amino acid sequence of protein A.
* **sequence_b:** Amino acid sequence of protein B.
* **label:** 0 or 1. 1 means affinity, 0 otherwise.

You need to modify the following parameters in utils/preprocess.py to generate dataset you want. 

* min_pos
* max_pos
* maxlen_a
* maxlen_b
* sample_ratio
* out_path

**For example**:
```python
out_dir = './data_ppi'
dataset_generator.generate_dataset_pipeline(min_pos=99,max_pos=100,maxlen_a=400,maxlen_b=400,sample_ratio=100,out_path=out_dir)
```
The code will proceed as follows:  
1. Find protein A whose number of positive samples in the STRING database is in the range of [min_pos,max_pos], and the length of the protein is less than maxlen_a.  
2. Make negative samples for each protein A, sample_ratio is the ratio of the negative positive sample, and the length of the negative sample is less than maxlen_b.
3. Generated trainset, testset, validation will be saved in outpath('./data_ppi')

#### **2. pre-calculate embeddings**
In order to speed up code training, embedding needs to be pre-calculated and saved.

**For example:**
```python
# Step2: Get embeddings and save as h5 file
    prot_seq_path = dataset_generator.prot_seq_path
    preprocessor = EmbeddingProcessor(prot_seq_path = prot_seq_path)
    preprocessor.generate_embeddings(out_dir=os.path.join(out_dir,'protein_embeddings'))
```
Protein embeddings will be saved in:   ./data_ppi/protein_embeddings

### Step2: Train the model
**Note:** First modify the train_path, test_path, val_path,embeddings_dir of args in the  **finetune_ddp.py** according to the path of the dataset obtained in step1: preprocess.

```python
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 finetune_ddp.py
```

### Step3: Test the model
**Note:** First modify the test_path,embeddings_dir and resume(model ckpt) of args in the  **evaluation.py**
```python 
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 evaluation.py
```

## Task2: Antibact peptide:training for experiments
**运行指令见quick.sh**
为了腾出空间，部分旧的模型ckpt存在了ssd1/zhangwt里面

### Step1: Classfication:
目前分类各阶段的实验结果：
|Step| Precision | Recall | Acc | F1_score|正负样本数量|
|:----:| :----: | :----: | :----: | :----: |:----:|
|Step1:6W Pretrain|0.905 | 0.882|0.968|0.893|11321:59728|    
|Step2:阴性菌finetune|0.943|0.959|0.978|0.951|1186:5898|
|Step3:鲍曼菌finetune|0.984|0.984|0.994|0.984|148:740|
|Step3:(Only step3)|0.906|0.951|0.975|0.928|148:740|
|Step3:(Only step1+3)|0.951|0.951|0.983|0.951|148:740|
|Step3:(Only step2+3)|0.968|1.0|0.994|0.984|148:740|

---

#### **下面Step3的数据比例按照train:test:val = 6:2:2划分**
|Step| Precision | Recall | Acc | F1_score|正负样本数量|
|:----:| :----: | :----: | :----: | :----: |:----:|
|Step3:鲍曼菌finetune|0.886|**0.969**|0.996|0.925|148:5898|
|Step3:(Only step3)|0.933|0.875|0.995|0.903|148:5898|
|Step3:(Only step1+3)|0.861|0.969|0.995|0.912|148:5898|
|Step3:(Only step2+3)|**0.968**|0.938|**0.998**|**0.952**|148:5898|

* **使用148：5898的数据，step2+step3的结果**

### Step2: Ranking:
1. **数据划分规则:** 将所有的肽(正样本m个，负样本n个)，根据某种比例划分到训练集，测试集，验证集中，再对训练集中的肽进行两两配对(正负，正正)，变成最终的训练集，这样划分之后，测试集，验证集中出现的所有肽，模型训练时都没有“见过”。  
2. **指标说明:**
实验结果中的topK_Precision = 模型预测的topK占真实topK的比例,**其中step2只需看到top50，因为测试集就50个正样本左右**
3. **模型:** 使用原始ESM的embedding，不改变embedding，只训练后面的MLP，模型输入pairs(a和b)，输出a > b的概率。(方法记为Pretrain-based MLP)   

结论和后续可尝试方案：
* 在无负负的情况下，LR低一点比较好，防止过拟合，step1 以后用lr=1e-7,step2用1e-6
* 目前的问题是容易混入很多的负样本，尚未确定加入负负是否比较好
* label可以尝试设置成，a比b好多少这种

#### Step1(使用P. 训练)实验结果：  

|Method|top10_Precision|top50_Precision|top100_Precision|top200_Precision|数据集划分比例|正负样本数量|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|Pretrain-based MLP(lr=1e-4,opt=Adam)|0.10|0.14|0.16|0.20|0.6:0.2:0.2|1186:5898|
|Pretrain-based MLP(lr=1e-5,opt=Adamw)|0|0.18|0.20|0.215|0.6:0.2:0.2|1186:5898|
|Pretrain-based MLP(lr=1e-6,opt=Adamw)|0|0.34|0.44|0.40|0.6:0.2:0.2|1186:5898|
|Pretrain-based MLP(lr=1e-7,opt=Adamw)|0|0.38|0.54|0.515|0.6:0.2:0.2|1186:5898|
|XGBoost|0.10|0.46|0.71|0.9|0.6:0.2:0.2|1186:5898|

#### Step2(鲍曼菌)实验结果：  
<!-- |Method|top10_Precision|top20_Precision|top40_Precision|数据集划分比例|正负样本数量|
|:----:| :----: | :----:|:----:|:----:|:----:|
|Pretrain-based MLP(from step1,lr=1e-4,opt=Adam)|0.0|0.10|0.05|0.4:0.3:0.3|148:5898|
|Pretrain-based MLP(无Step1,lr=1e-4,opt=Adam)|0.1|0.2|0.2|0.4:0.3:0.3|148:5898|
|Pretrain-based MLP(无Step1,lr=1e-5,opt=Adam)|0.4|0.3|0.3|0.4:0.3:0.3|148:5898|
|Pretrain-based MLP(无Step1,lr=1e-6,opt=Adam)|0.5|0.45|0.475|0.4:0.3:0.3|148:5898|
|Pretrain-based MLP(无Step1,lr=1e-5,opt=Adamw)|0.5|0.5|0.45|0.4:0.3:0.3|148:5898|
|Pretrain-based MLP(无Step1,lr=1e-6,opt=Adamw)|0.6|0.5|0.425|0.4:0.3:0.3|148:5898|
|XGBoost|0.40|0.50|0.78|0.4:0.3:0.3|148:5898| -->


|Method|top10_Precision|top20_Precision|top40_Precision|数据集划分比例|正负样本数量|
|:----:| :----: | :----:|:----:|:----:|:----:|
|Pretrain-based MLP (无Step1,lr=1e-6,opt=Adamw)|0.6|0.5|0.425|0.6:0.2:0.2|148:5898|
|Pretrain-based MLP (lr=1e-7,opt=Adamw)|0.2|0.50|0.375|0.6:0.2:0.2|148:5898|
|XGBoost|0.40|0.50|0.78|0.6:0.2:0.2|148:5898|

* 考虑到之前的模型预测topK中含有大量的负样本，认为模型在训练时没有处理负样本的能力(之前的数据只有正负，正正)。**下面的数据加入了负负数据，模型预测能力有显著提升**  

> 使用esm的参数为ranking模型生成embedding
> 

##### Step1(使用P. 训练)实验结果：
|Method|Top10_Prec|Top50_Prec|Top100_Prec|Top200_Prec|Ratio|P:N|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|Pretrain-based MLP |0.30|0.48|0.65|0.865|0.6:0.2:0.2|1186:5898|
|XGBoost|0.10|0.46|0.71|0.9|0.6:0.2:0.2|1186:5898|

##### Step2(鲍曼菌)实验结果：

|Method|Top10_Prec|Top20_Prec|Top40_Prec|Ratio|P:N|
|:----:| :----: | :----:|:----:|:----:|:----:|
|Pretrain-based MLP |0.60|0.75|0.875|0.4:0.3:0.3|148:5898|
|Pretrain-based MLP (无Step1)|0.60|0.55|0.85|0.4:0.3:0.3|148:5898|
|XGBoost|0.40|0.50|0.78|0.4:0.3:0.3|148:5898|

> 下面使用分类step1(p菌训练)的参数为ranking模型生成embedding
> 

##### Step1(使用P. 训练)实验结果：
|Method|Top10_Prec|Top50_Prec|Top100_Prec|Top200_Prec|Ratio|P:N|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|Pretrain-based MLP |0.40|0.50|0.70|0.885|0.6:0.2:0.2|1186:5898|
|XGBoost|0.10|0.46|0.71|0.9|0.6:0.2:0.2|1186:5898|


##### Step2(鲍曼菌)实验结果：

|Method|Top10_Prec|Top20_Prec|Top40_Prec|Ratio|P:N|
|:----:| :----: | :----:|:----:|:----:|:----:|
|Pretrain-based MLP|0.50|0.80|0.93|0.4:0.3:0.3|148:5898|
|Pretrain-based MLP (无Step1)|0.60|0.75|0.90|0.4:0.3:0.3|148:5898|
|XGBoost|0.40|0.50|0.78|0.4:0.3:0.3|148:5898|

**之前训的模型预测出来的肽亲疏水结构比较差，于是初步加入了结构信息**  

* **top20好坏比例：** 指regression/ranking出来的top20名中亲疏水性好的肽和亲疏水性差的肽的比例
* 下面的实验结果均是在训练:测试 = 7：3的数据上完成的（因为用在最终的预测任务，把验证集和训练集合并了）  
* 之前xgboost在金葡数据上ranking得到的top20好坏比例为13:7

|Method|top10_Precision|top20_Precision|top40_Precision|top20好坏比例(ranking)|top20好坏比例(regression)|
|:----:| :----: | :----:|:----:|:----:|:----:|
|无结构信息(原方法)|0.50|0.70|0.93|7:13|4:16|
|结构信息直接concate|0.60|0.75|0.95|5:15|9:11|
|结构信息+DAE(only step2)|0.30|0.70|0.85|13:7|**16:4**|
|结构信息+DAE|0.50|0.70|0.95|**14:6**|9:11|

**使用了基于lambdarank的框架训练(待更新)**:
##### Step1(使用P. 训练)实验结果(目前lr=1e-5更好，lr=1e-4在跑)：
|Method|Top10_Prec|Top50_Prec|Top100_Prec|Top200_Prec|Ratio|P:N|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|Pretrain-based MLP |0.40|0.50|0.70|0.885|0.6:0.2:0.2|1186:5898|
|XGBoost|0.10|0.46|0.71|0.9|0.6:0.2:0.2|1186:5898|


##### Step2(鲍曼菌)实验结果：

|Method|Top10_Prec|Top20_Prec|Top40_Prec|Ratio|P:N|
|:----:| :----: | :----:|:----:|:----:|:----:|
|Pretrain-based MLP|0.50|0.80|0.93|0.4:0.3:0.3|148:5898|
|Pretrain-based MLP (无Step1)|0.60|0.75|0.90|0.4:0.3:0.3|148:5898|
|XGBoost|0.40|0.50|0.78|0.4:0.3:0.3|148:5898|

### Step3: Regression:
注：
* 下面的Step1用了分类模型第一步训练后的参数作为模型的初始参数
* 下面MSE的统计基于MIC取了log10之后的值  

|Step| MSE |top10_MSE|top20_MSE|top60_MSE|pos_MSE|正负样本数量|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|Step1:使用P.训练|0.095|0.990|0.644|0.517|0.522|1186:5898|
|Step2:使用鲍曼菌训练|0.193|0.900|0.848|0.283|1.055|148:735|
|Step2:(only step2)|0.100|0.702|0.439|0.147|0.543|148:735|
|Step2:使用鲍曼菌训练|0.023|0.219|0.634|0.211|0.336|148:5898|
|Step2:(only step2)|0.040|1.048|1.205|0.402|1.201|148:5898|
---
* 不用分类第一步的参数作为模型初始参数的实验情况 

|Step| MSE |top10_MSE|top20_MSE|top60_MSE|pos_MSE|正负样本数量|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|Step1:使用P.训练|0.1338|1.129|1.105|0.765|0.751|1186:5898|
|Step2:使用鲍曼菌训练|0.065|0.275|0.287|0.096|0.407|148:735|
|Step2:(only step2)|0.161|0.350|0.701|0.235|0.643|148:735|
|Step2:使用鲍曼菌训练|0.0263|0.820|0.779|0.260|0.863|148:5898|    
|Step2:(only step2)|0.056|1.280|1.672|0.558|1.466|148:5898|  
---
* 用分类第二步(此处是指分类跳过Step1的情况)的参数作为模型初始参数的实验情况  

|Step| MSE |top10_MSE|top20_MSE|top60_MSE|pos_MSE|正负样本数量|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|Step1:使用P.训练|0.090|0.475|0.310|0.293|0.416|1186:5898|
|Step2:使用鲍曼菌训练|**0.006**|**0.095**|**0.173**|**0.058**|**0.213**|148:5898|
|Step2:(only step2)|0.034|1.112|1.011|0.337|1.189|148:5898|


在regression中也加入了结构信息，实验结果如下:  
* 选参数的标准是top30

|Step & Method| MSE |top10_MSE|top20_MSE|top30_MSE|top100_mse|pos_MSE|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|原方法(step1)|0.10|0.71|0.434|-|0.412|0.530|
|加入DAE的结构信息(step1)|0.10|0.851|-|0.482|0.375|0.504|
|原方法(step2)|0.005|0.168|0.153|0.102|-|0.212|
|加入DAE的结构信息(step2)|0.007|0.181|0.204|0.136|-|0.282|

<!-- * 选参数的标准是top30_mse
|Step & Method| MSE |top10_MSE|top20_MSE|top30_MSE|pos_MSE|top20好坏比例(regression)|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|只有ranking用结构信息，regression不用|0.005|0.168|0.153|0.102|0.212|10:10|
|ranking和reg都用结构信息|0.007|0.160|0.220|0.146|0.308|11:9| -->

在regression中也加入了结构信息，实验结果和结论如下:   
* 1. regression训练时，选模型参数的标准为top20_mse最小
* 2. 当regression不用结构信息时，pipeline出来的top7多肽结构特性都特别差，加入结构化信息之后有明显改善

|Step & Method| MSE |top10_MSE|top20_MSE|top30_MSE|pos_MSE|top20好坏比例(regression)|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|只有ranking用结构信息，reg不用|0.005|0.168|0.153|0.102|0.212|10:10|
|ranking和reg都用结构信息|0.007|0.181|0.204|0.136|0.282|12:8|


## Task2: Antibact peptide: training for predictions
**六肽Pipeline** ：
* Step1: 分类: 用P菌训练，数据1186:5898, train:test = 9:1
* Step2: 分类: 用鲍曼菌finetune，数据148:5898, train:test = 8:2
* Step3: 排序: P菌训练(embedding使用来自分类step1(P菌)的参数)，数据1186:5898,train:test = 8:2
* Step4: 排序: 鲍曼菌finetune，数据148:5898, train:test = 7:3
* Step5: 回归: P菌训练(embedding使用分类Step1（P菌）的参数)，数据1186:5898, train:test = 9:1
* Step6: 回归: 鲍曼菌finetune，数据148:5898, train:test = 8:2

### 1. Classification:
|Step| Precision | Recall | Acc | F1_score|正负样本数量|
|:----:| :----: | :----: | :----: | :----: |:----:|   
|Step1:P.菌finetune|0.941|0.936|0.970|0.936|1186:5898|
|Step2:鲍曼菌finetune|1.0|0.969|0.999|0.984|148:5898|

### 2. Ranking
* Step1:   

|Method|top10_Precision|top50_Precision|top100_Precision|top200_Precision|数据集划分比例|正负样本数量|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|Pretrain-based MLP |0.40|0.68|0.77|0.930|0.8:0.2|148:5898|

* Step2: 

|Method|top10_Precision|top20_Precision|top40_Precision|数据集划分比例|正负样本数量|
|:----:| :----: | :----:|:----:|:----:|:----:|
|Pretrain-based MLP|0.50|0.70|0.93|0.7:0.3|148:5898|

### 3. Regression

|Step| MSE |top10_MSE|top20_MSE|top60_MSE|pos_MSE|正负样本数量|
|:----:| :----: | :----:|:----:|:----:|:----:|:----:|
|Step1:P.菌finetune|0.090|0.693|0.414|0.382|0.456|1186:5898|
|Step2:鲍曼菌finetune|0.011|0.119|0.35|0.117|0.360|148:5898|

### Pipeline时间估计:
注:按照最坏结果估计，实际耗时小于估计耗时
#### 1.6400w搜索:
* Step1分类:178h,剩余640w条肽
* Step2排序:预先计算embedding 17h + 排序5h。合计22h，剩余500条
* Step3回归:1min左右？
合计：8.3天

#### 2.393w搜索:
* Step1分类:10h,剩余39w条肽
* Step2排序:预先计算embedding 1h + 排序0.5h。合计1.5h，剩余500条
* Step3回归:1min左右？
合计：0.5天

### 设计到结构化信息的文件说明
1. 用few-shot-peptides的cal_pep_des
    * 为指定的(1w正+1w负)肽生成结构化数据(norm后，676维)，存在/antibact_prediction/structured_data_norm/data4daetraining
    * csv文件存在data_antibact/final_data/pretrain_based/structured_data_csv/下，原版数据为data4daetraining(2w).csv，norm后的为data4dae_training_norm.csv
2. 使用antibact_preprocess.py为ranking和reg生成encodings：存储在：./data_antibact/final_data/pretrain_based/dae_encodings    
2. 使用antibact_preprocess.py为final prediction生成12w的encodings存储:/antibact_prediction/dae_encodings

3. 用antibact_preprocess.py 为ranking训练的结构化数据做归一化(2w数据的参数)，676维存在 ./data_antibact/final_data/pretrain_based/normed_structured_data

* 加入tabnet之后
1. 为了训一个393w数据的无监督tabnet，使用few-shot-peptide/generate_sub_dataset.py为393w的数据做了一个筛选，只保留结构相关的81维的特征，筛选之后的结果存放在data_antibact/final_data/pretrain_based/structured_data_csv/393w_spefeatures.csv
2. tabnet训练的时候对393w的数据做了预处理(不过这里相当于没动，因为所有的都是数值型特征，只是填充了一下缺失值)，然后分成了3个数据集，存放在/ssd1/zhangwt/DrugAI/projects/esm/data_antibact/final_data/pretrain_based/data4tabpretrain/393w下面
3. 使用antibact-preprocess 为pipeline的ranking和reg生成encodings
   1. 训练用，存放在：/data_antibact/final_data/pretrain_based/tabnet_embeddings和/tabnet_orifeatures里，分别对应tabnet的8维编码和原始81维数据特征
   2. 12w数据，prediction用，存放在：/antibact_prediction/tabnet_orifeatures里，对应81维数据特征
4. ddd


### 目前最好的结果：
非多模态:  
lambdarank.py 使用random sample跑出来的结果，ckpt存在lambdarank_randomsap_exp里面，pipeline结果在antibact_prediciton_new 里面
多模态:  
lambdarank_multimodal.py pipeline结果在antibact_prediciton_mm 里面


### 2023.2.7 关于7，8肽的pipeline实验
```bash
sh pipeline.sh
```
6，7，8肽的实验结果在prediction文件夹中  

#### 步骤
1. 把数据分块多卡跑分类
2. 正样本整合再生成结构化数据
3. 完成排序和回归

#### 时间估计
* 分类任务单卡每秒45条序列
* 结构化数据生成单核40条序列
  
1. 7肽分类单卡需要413小时，三张卡，6天可以完成
2. 8肽分类单卡需要6000小时，四张卡，需要60天