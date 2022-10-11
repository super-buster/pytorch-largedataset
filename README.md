# pytorch-largedataset
this project provides a demonstration of how to load large files in pytorch dataset.
Basically, we can rely on line offset and mmmap tech. Here are examples:
```
from pytorch_largedataset import LargeDataset, LargeIterableDataset, DataFiles
large_file_path='train.txt'
train_offset_path='train_offset'
data_files=DataFiles(large_file_path,train_offset_path)
train_set=LargeIterableDataset(data_files)
for example in train_set:
    print(example)
```
For a benchmark comparing two modes, you can check benchmark.py file.
