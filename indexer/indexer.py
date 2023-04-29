import sys
import resource
import argparse
import multiprocessing
import json,os,glob,psutil,gc
import time
from Parser import Parser
from disk_mergesort import disk_mergesort
from encoding import encode, read_encoded_line, vbyte_encoding

MEGABYTE = 1024 * 1024
def memory_limit(value):
    limit = value * MEGABYTE
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

#function to create a generator from jsonlines file (low memory consumption)
def document_reader(json_doc):
    with open(json_doc,'r',encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)

#class responsible for index creation
class IndexCreator:
    def __init__(self,n_threads,mem_limit,save_path):
        self.n_threads=n_threads
        manager = multiprocessing.Manager()
        self.file_lock = manager.Lock()
        self.variable_lock = manager.Lock()
        self.queue = manager.Queue(maxsize=n_threads)
        self.flush_count = multiprocessing.Value('i', 0)
        self.did = multiprocessing.Value('i', 0)
        self.total_documents = multiprocessing.Value('i', 0)
        self.avg_len = multiprocessing.Value('i', 0)
        self.mem_limit = mem_limit
        self.save_path = save_path
        self.parser= Parser()

        
    def create(self,corpus_generator):
        #function to read n values from generator(avoid high memory usage)
        def readN(generator,n):
            aux = []
            try:
                while len(aux)<n:
                    aux.append(next(generator))
            except StopIteration:
                pass
            self.total_documents.value += len(aux)
            return aux
        
        limit = 1000
        threads=[]
        for _ in range(self.n_threads):
            t = multiprocessing.Process(target=self.create_parallel)
            t.start()
            threads.append(t)
        #add documents to q (l for each process)
        while True:
            next_lines = readN(corpus_generator,limit)
            if next_lines==[]:
                break
            self.queue.put(next_lines)
        for _ in range(self.n_threads):
            self.queue.put([])

        for t in threads:
            t.join()
        self.save_doc_index_and_statistics()
            
    def create_parallel(self): 
        document_index={}
        index={}
        while True:
            #read corpus chunk
            corpus = self.queue.get(block=True)
            if len(corpus)==0:
                self.dump(index,document_index)
                del index,document_index
                gc.collect()
                break

            #check memory requirements
            memory_used=psutil.Process(os.getpid()).memory_info().rss
            if memory_used >= ((0.9 * self.mem_limit * MEGABYTE)/self.n_threads):
                self.dump(index,document_index)
                del index,document_index
                gc.collect()
                index={}
                document_index={}

            #add documents to index
            for document in corpus:
                self.variable_lock.acquire()
                did = self.did.value
                self.did.value+=1
                self.variable_lock.release()
                
                length,tokens = self.parser.freq_dist(' '.join([document['title'],document['text']]+document['keywords']))
                document_index[did] = [document['id'],length]
                for i,(term,tf) in enumerate(tokens):
                        if term not in index.keys():
                            index[term] = []
                        index[term].append((did,tf))
            del corpus
        
    def save_doc_index_and_statistics(self):
        with open(f'{self.save_path}/corpus_statistics','w') as corpus_stats:
            corpus_stats.write(str(self.total_documents.value)+','+str(float(self.avg_len.value)/self.total_documents.value)+'\n')

    #dump index to memory
    def dump(self,index,doc_index):
        with open(f'{self.save_path}/{self.flush_count.value}.temp','wb') as index_file:
            self.flush_count.value+=1
            for term in sorted(index):
                inverted_list=[]
                #find delta gaps
                previous_did=0
                for o in index[term]:
                    inverted_list.append((o[0]-previous_did,o[1]))
                    previous_did=o[0]
                index_file.write(encode(term,inverted_list))
        
        self.file_lock.acquire()
        with open(f'{self.save_path}/document_index','a',encoding='utf-8') as file:
            for did in doc_index:
                did_attributes=doc_index[did]
                file.write(str(did)+','+str(did_attributes[0])+','+str(did_attributes[1])+'\n')
                self.avg_len.value+= did_attributes[1]
        self.file_lock.release()
    
    #merges all partial index created
    def merge_indexes(self):
        temps = glob.glob(f'{self.save_path}/*.temp')
        while len(temps)>1:
            #generate tuples of different files (e.g. file1.txt,file2.txt)
            it=iter(temps)
            f_tuples = list(zip(it,it))

            #create arguments for disk mergesort parallel executions
            args = [(tup[0],tup[1],tup[0]+tup[1].replace(f'{self.save_path}/','').replace(f'{self.save_path}','')) for tup in f_tuples]

            #create pool with maximum n_threads processes (but may be less if there are less tasks)
            with multiprocessing.Pool(min([self.n_threads,len(f_tuples)])) as pool:
                pool.starmap(func=disk_mergesort,iterable=args)

            temps = glob.glob(f'{self.save_path}/*.temp')
        os.rename(temps[0],f'{self.save_path}/invertedindex.txt')
    
    def create_lexicon(self):
        lex = {}
        #get information from inverted index
        with open(f'{self.save_path}/invertedindex.txt','rb') as index:
            while True:
                #save position of term in file to lexicon
                offset = index.tell()
                term,inv_list = read_encoded_line(index)
                if term is None:
                    break
                lex[term] = (offset,len(inv_list))

        #write to lexicon and save important information in ram
        self.avg_list_size = 0
        self.number_of_lists = 0
        with open(f'{self.save_path}/lexicon','w',encoding='utf-8') as file:
            for term in lex:
                offset,list_len = lex[term]
                self.avg_list_size+=list_len
                self.number_of_lists += 1
                file.write(term)
                file.write(',')
                file.write(str(offset))
                file.write(',')
                file.write(str(list_len))
                file.write('\n')
        self.avg_list_size = float(self.avg_list_size)/self.number_of_lists

    #reads index to memory (it reads entire index, so very memory intensive)
    def read_index_file(self,path):
        index={}
        with open(path) as index_file:
            while True:
                term,line = read_encoded_line(index_file)
                if term is None:
                    break
                index[term] = line
        return index

def main(args):
    """
    Your main calls should be added here
    """
    start_time = time.time()
    index=IndexCreator(n_threads=psutil.cpu_count(logical=False),mem_limit=args.memory_limit,save_path=args.index)
    corpus_generator = document_reader(args.corpus)
    index.create(corpus_generator)
    index.merge_indexes()
    index.create_lexicon()
    elapsed_time = time.time()-start_time
    stats = {}
    stats['Elapsed Time'] = round(elapsed_time)
    stats['Index Size'] = round(os.path.getsize(f'{args.index}/invertedindex.txt') / (1024 * 1024),2) #size in mb
    stats['Number of Lists'] = index.number_of_lists
    stats['Average List Size'] = round(index.avg_list_size,2)
    print(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '-m',
        dest='memory_limit',
        action='store',
        required=True,
        type=int,
        help='memory available'
    )
    parser.add_argument(
        '-c',
        dest='corpus',
        action='store',
        required=True,
        type=str,
        help='the path to the corpus file to be indexed.'
    )
    parser.add_argument(
        '-i',
        dest='index',
        action='store',
        required=True,
        type=str,
        help='the path to the directory where indexes should be written'
    )

    args = parser.parse_args()
    memory_limit(args.memory_limit)
    try:
        main(args)
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)


# You CAN (and MUST) FREELY EDIT this file (add libraries, arguments, functions and calls) to implement your indexer
# However, you should respect the memory limitation mechanism and guarantee
# it works correctly with your implementation