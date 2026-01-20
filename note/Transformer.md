# 基础概念
## 发展历程
1. 词向量
   向量空间模型（VSM），每个维度代表一个特征项（字，词等）的权重，通常由TF和IDF来计算
2. 语言模型
   N-gram模型的核心思想是基于马尔可夫假设，即一个词的出现概率仅依赖于它前面的N-1个词。`但当N较大时，会出现数据稀疏性问题。模型的参数空间会急剧增大`
3. Word2Vec
   * 连续词袋模型CBOW(Continuous Bag of Words)是根据目标词上下文中的词对应的词向量, 计算并输出目标词的向量表示；
   * Skip-Gram模型, 是利用目标词的向量表示计算上下文中的词向量. 实践验证CBOW适用于小型数据集, 而Skip-Gram在大型语料中表现更好。
4. ELMo
   （Embeddings from Language Models）首先在大型语料库上训练语言模型，得到词向量模型，然后在特定任务上对模型进行微调，得到更适合该任务的词向量，ELMo首次将预训练思想引入到词向量的生成中，使用双向LSTM结构，能够捕捉到词汇的上下文信息，生成更加丰富和准确的词向量表示
# Transformer架构
## 注意力机制
### 背景
   RNN 及 LSTM 虽然具有捕捉时序信息、适合序列生成的优点，但有两个缺陷：
   * 限制了并行计算的能力
   * 难捕捉长序列相关关系，需要将整个序列读入内存以此计算
   注意力机制源于cv，有三个核心变量：Query、Key、Value。通过对Query和Key运算可以得到权重，反映了从Query触发，对文本每一个token应该分布的注意力相对大小，通过权重和Value运行，可以得到从Query触发计算整个文本注意力的结果。
### 深入理解注意力
   通过词向量的欧氏距离来衡量词向量的相似性，同样也可以用点积来度量，语义相似则点积大于0，否则小于0。
   假设有Query“fruit”的词向量q，Key的词向量$k=[v_{apple}, v_{banana}, v_{chair}]$ ，那q和每个键值的的相似程度：$$x=qK^T$$ x是一个维度为$d_k$的向量，即和Key的数量一致，然后利用Softmax层进行归一化，反应的是Query和每一个Key的相似程度，注意力分数和值向量相乘即可。
   同时一次性查询多个Query，将多个Query堆叠在一起形成矩阵Q，同时如果Q和K的维度$d_k$非常大的话，softmax放缩容易受影响，要除以一个词向量维度$\sqrt{d_k}$有$$attenion(Q,K,V)=softmax(QK^T)V$$
### 注意力机制实现
```python
def attention(query, key, value, dropout=None):
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	# Softmax
	p_attn = scores.softmax(dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn
```
### 自注意力
实际应用往往只需要Q和K的注意力结果，在Transformer的Encoder结构中，使用的是注意力的变种——自注意力机制。所谓自注意力，是计算本身序列中每个元素对其它元素的注意力分布，在计算过程中，Q，K，V都由同一个输入通过不同的参数矩阵计算得到，在Encoder中，Q，K，V分别是输入对参数矩阵$W_q,W_k,W_v$做积得到
### 掩码自注意力
掩码遮蔽特定位置的token，学习过程中会忽略这些token。
* 核心动机：让模型只能使用历史信息进行预测，而看不到未来信息。使用注意力机制的Transformer模型也是通过类似于n-gram的语言模型任务来学习的，也就是对一个文本序列，不断根据之前的token来预测下一个token，直到将整个文本序列补全。
```
<BOS> 【MASK】【MASK】【MASK】【MASK】
<BOS>    I   【MASK】【MASK】【MASK】
<BOS>    I     like  【MASK】【MASK】
<BOS>    I     like    you  【MASK】
<BOS>    I     like    you   </EOS>
```
掩码矩阵是一个上三角矩阵，对于输入维度为(batch_size, seq_len, hidden_size)时，mask矩阵一般为(1, seq_len, seq_len)
* 实现
```python
# 创建上三角矩阵
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
# triu用于创建一个上三角矩阵
mask = torch.triu(mask, diagonal=1)
# 取出需要的序列长度
scores = scores + mask[:, :seqlen, :seqlen]
# 将注意力分数与掩码求和，在做Softmax操作
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```
### 多头注意力
单一的注意力机制很难全面拟合语句序列里的相关关系，将最后的多次结果拼接起来作为最后的输出，即可更全面深入地拟合语言信息。
所谓的多头注意力机制其实就是将原始的输入序列进行多组的自注意力处理；然后再将每一组得到的自注意力结果拼接起来，再通过一个线性层进行处理，得到最终的输出。实现代码：
```python
import torch.nn as nn
import torch

# 多头自注意力
class MultiHeadAttention(nn.Module):
	def __init__()
	# 隐藏层维度必须是头数整数倍，因为需要把输入拆成头数个矩阵
	assert args.dim % args.n_heads == 0
	self.head_dim = args.dim // args.n_heads
	self.n_heads = args.n_heads
	# Wq,Wk,Wv参数矩阵，维度为n_embd, dim
	# 通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接等同于拼接矩阵再内积
	self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
	self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
	self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
	# 输出注意力矩阵，维度为dim x dim (head_dim = dim/n_heads)
	self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
	self.attn_dropout = nn.Dropout(args.dropout)
	self.resid_dropout = nn.Dropout(args.dropout)
	self.is_causal = is_causal
	# 创建一个上三角矩阵
	if is_causal:
		mask = torch.full((1,1, args.max_seq_len, args.max_seq_len), float("-inf"))
		mask = torch.triu(mask, diagonal=1)
		self.register_buffer("mask", mask)

	def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):
		bsz, seqlen, _ = q.shape
		# 计算qkv，输入通过参数矩阵层
		# (B, T, n_head, dim//n_head),然后交换维度，变成(B, n_head, T, dim//n_head)
		xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
		xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
		xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
		xq = xq.transpose(1, 2)
		xk = xk.transpose(1, 2)
		xv = xv.transpose(1, 2)
		# 计算QK^T/sqrt(d_k)
		# 维度为(B, n_heads, T, h_dim)*(B, n_heads, h_dim, T)
		# =(B, n_heads, T, T)
		scores = torch.matmul(xq, xk,transpose(2, 3)) / math.sqrt(self.head_dim)
		if self.is_casual:
			assert hasattr(self, 'mask')
			scores = scores + self.mask[:, :, :seqlen, :seqlen]
		scores = F.softmax(scores.float(), dim=-1).type_as(xq)
		scores = self.attn_dropout(scores)
		# V * Scores, 维度为(B, n_heads, T, h_dim)
		output = torch.matmul(scores, xv)
		# 交换维度，恢复时间维度，再拼接
		# 交换为(B, T, n_heads, h_dim)，再拼接(B, T, dim)
		# contiguous()开辟新内存，直接transpose再view会出错
		output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
		# 投影回残差流
		output = self.wo(output)
		output = self.resid_dropout(output)
		return output
```
## Encoder-Decoder
Transformer由Encoder和Decoder组成，每个Encoder/Decoder由6个Layer组成
### 前馈神经网络（Feed Forward Neural Network）
```python
class MLP(nn.Module):
	'''前馈神经网络'''
	def __init__(self, dim:int, hidden_dim:int, dropout:float):
		super().__init__()
		# 第一层线性变换，输入维度到隐藏维度，一般升4倍
		self.w1 = nn.Linear(dim, hidden_dim, bias=False)
		# self.relu = F.relu(x)         # 标准Transformer非线性激活
		# 第二层线性变换，隐藏维度到输入维度
		self.w2 = nn.Linear(hidden_dim, dim, bias=False)
		# dropout层，防止过拟合
		self.dropout = nn.Dropout(dropout)
	def forward(self, x):
		# 前向传播函数
		# 首先，输入x通过第一层线性变换和RELU激活函数
		x = self.w1(x)
		x = self.relu(x)
		x = self.w2(x)
		x = self.dropout(x)
		# 最后，通过第二层线性变换和dropout
		return x
```
### 层归一化
神经网络主流的归一化一般有两种，批归一化（Batch Norm）和层归一化（Layer Norm）
1. 批归一化：对每个样本的值减去均值再除以标准差来将这一个 mini-batch 的样本的分布转化为标准正态分布。
缺陷：
- 当显存有限，mini-batch 较小时，Batch Norm 取的样本的均值和方差不能反映全局的统计分布信息，从而导致效果变差；
- 对于在时间维度展开的 RNN，不同句子的同一分布（同一位置的token）大概率不同，所以 Batch Norm 的归一化会失去意义；
- 在训练时，Batch Norm 需要保存每个 step 的统计信息（均值和方差）。在测试时，由于变长句子的特性，测试集可能出现比训练集更长的句子，所以对于后面位置的 step，是没有训练的统计量使用的；
- 应用 Batch Norm，每个 step 都需要去保存和计算 batch 统计量，耗时又耗力，计算次数(T * dim)
相较于 Batch Norm 在每一层统计所有样本的均值和方差，Layer Norm 在每个样本上计算其所有层的均值和方差，从而使每个样本的分布达到稳定。Layer Norm 的归一化方式其实和 Batch Norm 是完全一样的，只是统计统计量的维度不同，计算次数是(bsz * T)
```python
class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        # 统计每个样本所有维度的值
        mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True) # std:[bsz, max_len, 1]
        output = self.a_2 * (x - mean) / (std + self.eps) +self.b_2
        return output
```
### 残差连接
在第一个子层，输入进入多头自注意力层的同时会直接传递到该层的输出，然后该层的输出会与原输入相加，再进行标准化。在第二个子层也是一样。
$$
x=x+MultiHeadSelfAttention(LayerNorm(x))\\
output=x+FNN(LayerNorm(x))
$$
代码实现中，通常在层的forward计算中加上原值来实现
```python
# 残差 + 注意力计算
h = x + self.attention.forward(self.attention_norm(x))
# 残差 + 前馈网络输出
output = h + self.feed_forward.forward(self.fnn_norm(h))
```
代码中 self.attention_norm 和 self.fnn_norm 都是 LayerNorm 层
### Encoder
Encoder 由 N 个 Encoder Layer 组成，每一个 Encoder Layer 包括一个注意力层和一个前馈神经网络。
```python
class EncoderLayer(nn.Module):
	def __init__(self, args):
		super().__init__()
		# 注意力层前层归一化
		self.attention_norm = LayerNorm(args.n_embd)
		# Encoder不需要掩码
		self.attention = MultiHeadAttention(args, is_casual=False)
		self.fnn_norm = LayerNorm(args.n_embd)
		# 将前馈网络输入、隐藏层维度设为dim
		self.feed_forward = MLP(args.dim, args.dim, args.dropout)

	def forward(self, x):
		# 注意力层->前馈神经网络层，每层前都对输入加层归一化LN，且输出都加上输入
		norm_x = self.attention_norm(x)
		# 自注意力
		attn = x + self.attention(norm_x)
		norm_attn = self.fnn_norm(attn)
		out = attn + self.feed_forward(norm_attn)
		return out

class Encoder(nn.Module):
	def __init__(self, args):
		super(Encoder, self).__init__()
		# n_layer个Encoder_Layer构成Encoder
		self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
		self.norm = LayerNorm(args.n_embd)
	def forward(self, x):
		# n层encoder layer后对输出层归一化
		for layer in self.layers:
			x = layer(x)
		return self.norm(x)
```
Encoder的输出就是对输入编码的结果
### Decoder

