import os
from tkinter import SEL_LAST
import h5py
from collections import namedtuple
import copy
import pandas as pd
import torch

Prot = namedtuple('Prot',['seq','emb','mic'])

class ranking_tool:
    """
    A module for top_K ranking, mainly implemented with max-heap
    """
    def __init__(self, model, device, embeddings_dir, max_num = 500):
        self.model = model
        self.device = device
        self.max_num = max_num
        self.cur_num = 0
        self.embeddings_dir = embeddings_dir
        self._elements = [Prot(seq=None,emb=None,mic=1e5)]*self.max_num

    def insert(self, Prot):
        """
        Add an element to heap
        """
        if self.cur_num < self.max_num:
            self._elements[self.cur_num] = Prot
            self.cur_num += 1
            return

        else:
            if self.compare(Prot,self.get_top()):
                self._elements[0] = Prot
                self._siftdown(0,self.cur_num)

    def get_top(self):
        return self._elements[0]

    def makeheap(self):
        i = int(self.cur_num / 2) - 1
        while i >= 0:
            self._siftdown(i,self.cur_num)
            i -= 1

    def isfull(self):
        return self.max_num == self.cur_num

    def _siftdown(self, idx, n):
        """
        Sift down the idx node.
        Args:
            idx: The node to siftdown
            n: Node num of the heap
        """
        temp = self._elements[idx]
        j = 2 * idx + 1
        while(j < n):
            # MIC of right child is bigger than left child
            if (j + 1 < n):
                if (self.compare(self._elements[j],self._elements[j+1])):
                    j += 1

            if self.compare(self._elements[j],temp):
                break

            self._elements[idx] = self._elements[j]
            idx = j
            j = 2 * idx + 1
        self._elements[idx] = temp

    def compare(self,prot_a,prot_b):
        """
        Return priority Prot_a > Prot_b.(Means the lower mic)
        """
        emb_a = prot_a.emb.to(self.device)
        emb_b = prot_b.emb.to(self.device)
        features = torch.cat([emb_a,emb_b,torch.abs(emb_a-emb_b)],dim=1)
        with torch.no_grad():
            logits = self.model(features)
            logits = torch.sigmoid(logits).cpu().numpy()

        return logits > 0.5

    # def compare(self,prot_a,prot_b):
    #     """
    #     Return priority Prot_a > Prot_b.(Means the lower mic)
    #     """     
    #     u = prot_a.emb.to(self.device)
    #     v = prot_b.emb.to(self.device)
    #     with torch.no_grad():
    #         logits = self.model(u,v)
    #         logits = torch.sigmoid(logits).cpu().numpy()
    #     return logits > 0.5

    def _res2dataframe(self):
        """
        Convert result to dataframe
        """
        seqs_list = []
        mic_list = []
        for i in self._elements:
            seqs_list.append(i.seq)
            mic_list.append(i.mic)

        return pd.DataFrame({'sequence':seqs_list,'MIC':mic_list})

    def get_result(self):
        """
        Return the top K result
        """
        i = self.cur_num - 1
        while(i >= 0):
            self._elements[0],self._elements[i] = self._elements[i],self._elements[0]
            self._siftdown(0,i)
            i -= 1
        
        result_df = self._res2dataframe()
        return result_df

    def reset(self):
        """
        Reset the heap
        """
        self.model.eval()
        self.cur_num = 0
        self._elements = [Prot(seq=None,emb=None,mic=1e5)]*self.max_num

    def __len__(self):
        return self.max_num

if __name__ == "__main__":
    
    rk = ranking_tool("test","hello","",max_num=10)
    for i in range(80,90):
        rk.insert(Prot("None","None",i))
        rk.makeheap()
    # result = rk.get_result()
    # print("first result is ",result)

    for i in range(0,5):
        rk.insert(Prot("None","None",i))
    # result = rk.get_result()
    # print("second result is ",result)

    for i in range(30,40):
        rk.insert(Prot("None","None",i))
    # result = rk.get_result()
    # print("third result is ",result)

    for i in range(5,22):
        rk.insert(Prot("None","None",i))
    result = rk.get_result()
    print("final result is ",result)    

    