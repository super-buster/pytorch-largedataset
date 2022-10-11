import dill as pickle
from timeit import timeit
from pytorch_largedataset import LargeDataset, LargeIterableDataset,DataFiles

large_file_path='train.txt'
train_offset_path='train_offset'
num_lines = 200000000

def generate_file():
    with open(large_file_path,"w") as f: 
        for _ in range(num_lines):
            f.write("This is really a large file!\n")

def benchmark_offset_seek(data_files):
    train_set=LargeDataset(data_files)
    def _iterateit():
        for example in train_set:
            pass
    print(f"benchmark offset seek consumes {timeit(_iterateit,number=3)}")

def benchmark_mmap(data_files):
    train_set=LargeIterableDataset(data_files)
    def _iterateit():
        for example in train_set:
            pass
    print(f"benchmark mmap consumes {timeit(_iterateit,number=3)}")

if __name__=='__main__':
    generate_file()
    offset=[]
    with open(large_file_path,'r') as f: 
        for lino in range(num_lines):
            if lino % 100000000 == 0:
                print("{} lines have been processed!".format(lino))
            offset.append(f.tell())
            f.readline()

    with open(train_offset_path,"wb") as trof: 
        pickle.dump(offset,trof)
    data_files=DataFiles(large_file_path,train_offset_path)
    benchmark_offset_seek(data_files)
    benchmark_mmap(data_files)
