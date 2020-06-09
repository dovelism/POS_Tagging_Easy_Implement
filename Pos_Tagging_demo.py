#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：POS_Tagging_Easy_Implement -> Pos_Tagging_demo
@Author ：Dovelism
@Date   ：2020/6/9 下午1:34
@Desc   ：
=================================================='''

import numpy as np

tag2id,id2tag = {},{}  # maps tag to id, tag2id:{"VB":0 , "NNP":1}
word2id,id2word = {},{}

for line in open('train_data.txt'):
    items = line.split('/')
    word,tag = items[0],items[1].rstrip() #抽取每一行的单词和词性
    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(id2tag)] = tag
M = len(word2id)     # M:词典大小
N = len(tag2id)      # N:词性的种类个数

#print(M,N)
#print(tag2id)

'''
    构建 pi,A,B
'''

pi = np.zeros(N) # pi[i]出现在第一个位置的概率
A = np.zeros((N,M)) # A[i][j]:给定tag i，出现单词j的概率
B = np.zeros((N,N)) # B[i][j]:之前的状态是i，之后转成状态j的概率

def log(v):
    if v == 0:
        return np.log(v+0.000001)
    return np.log(v)


prev_tag = ""
for line in open('train_data.txt'):
    items = line.split('/')
    wordID, tagID = word2id[items[0]], tag2id[items[1].rstrip()]
    if prev_tag == "":  # 句子开始
        pi[tagID] += 1
        A[tagID][wordID] += 1
    else:  # 不是句子开头
        A[tagID][wordID] += 1
        B[tag2id[prev_tag]][tagID] += 1
    if items[0] == ".":
        prev_tag = ""
    else:
        prev_tag = items[1].rstrip()

# normalize
pi = pi / sum(pi)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])


#print(pi)
#print(id2tag)

#print(B)


def viterbi(x, pi, A, B):
    """
    x：用户输入字符串或序列
    pi:initial probability of tags
    A: 给定tag，每个单词出现的概率
    B：tag之间的转移概率
    """
    x = [word2id[word] for word in x.split(" ")]  # x:[4532,334,523,335,..]
    T = len(x)

    dp = np.zeros((T, N))  # dp[i][j] : w1,...,wi,假设wi的tag是第j个tag
    pointer = np.array([[0 for x in range(N)] for y in range(T)])  # T*N

    for j in range(N):  # base case
        dp[0][j] = log(pi[j]) + log(A[j][x[0]])  # 第一列
    for i in range(1, T):  # 每个单词
        for j in range(N):  # 每个词性
            dp[i][j] = -99999
            for k in range(N):  # 从每一个k可以到达j
                score = dp[i - 1][k] + log(B[k][j]) + log(A[j][x[i]])  # 核心
                if score > dp[i][j]:
                    dp[i][j] = score
                    pointer[i][j] = k

    # 把最好的tag sequence找到
    best_seq = [0] * T  # best_seq=[1,3,5,23,4]
    # step1：找出对应于最后一个单词的词性
    best_seq[T - 1] = np.argmax(dp[T - 1])
    # step2:通过从后到前的循环来依次求出每个单词的词性
    for i in range(T - 2, -1, -1):
        best_seq[i] = pointer[i + 1][best_seq[i + 1]]
    # 到目前为止，best_seq存放了对应于x的词性序列

    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])




if __name__ == '__main__':

    x = "Social Security number , passport number and details about the services provided for the payment"
    viterbi(x, pi, A, B)