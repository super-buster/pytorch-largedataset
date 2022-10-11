import torch
import dill as pickle
from collections import namedtuple
from mmap import mmap
from torch.utils.data import IterableDataset,Dataset

DataFiles=namedtuple('DataFiles',['data_file','offset_file'])
DataFiles.__new__.__defaults__=(None,None)

# implement single sentence dataloader or sentence pair 
# you could refer to https://github.com/google-research/bert/blob/master/extract_features.py
class LargeDataset(Dataset):
    def __init__(self,data_files:DataFiles):
        super(LargeDataset).__init__()
        self.data_files=data_files
        print("loading offsets files... ")
        with open(data_files.offset_file,'rb') as f: 
            self.offsets=pickle.load(f)
        self.num_lines=len(self.offsets)
        print(f"{self.num_lines} lines")
    
    def __len__(self) -> int:
        return self.num_lines

    def __getitem__(self,index):
        offset=self.offsets[index]
        with open(self.data_files.data_file,'r',encoding='utf-8') as f: 
            f.seek(offset)
            line=f.readline()
            return line.strip()

    def _truncate_seq(self,tokens,max_length):
        raise NotImplementedError

    def get_input_features(self,example):
        raise NotImplementedError

    def convert_examples_to_features(self,example):
        raise NotImplementedError

class LargeIterableDataset(IterableDataset):
    def __init__(self,data_files: DataFiles):
        self.data_files=data_files
        print("mapping files to memory...")
        with open(data_files.data_file,"r+") as f: 
            self.mm=mmap(f.fileno(),0)
        
    def __iter__(self):
        start=0
        for end, bt in enumerate(self.mm):
            if bt==b"\n":
                yield self.mm[start:end].decode()
                start = end+1

    def _truncate_seq(self,tokens,max_length):
        raise NotImplementedError

    def get_input_features(self,example):
        raise NotImplementedError
        
    def convert_examples_to_features(self,example):
        raise NotImplementedError

