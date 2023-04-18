import sys
#import resource
import argparse

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import multiprocessing
from multiprocessing.pool import ThreadPool
from collections import deque
from struct import pack,unpack
import nltk
import string,json,os,glob,psutil,gc
import time

MEGABYTE = 1024 * 1024
def memory_limit(value):
    limit = value * MEGABYTE
    #resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

#postings lists to be read at once or saved at once
BUFFER_LEN = 5000 

#function to create a generator from jsonlines file (low memory consumption)
def document_reader(json_doc):
    with open(json_doc,'r',encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)

#encoding related functions below
#variable bytes encoding for a number
def vbyte_encoding(number):
    bytes_list = []
    while True:
        bytes_list.insert(0, number % 128)
        if number < 128:
            break
        number = number // 128
    bytes_list[-1] += 128
    return pack('%dB' % len(bytes_list), *bytes_list)

#encodes a "line" for saving to file
#comma(\x2c) separates term from inverted_list
#NULL(\x00) separates final number of inverted_list to other term,
#this works because a number can neither start nor end with NULL
def encode(term,line):
    try:
        b = bytearray(term,encoding='utf-8')
        b.extend(b'\x2c')
        for tup in line:
            b.extend(vbyte_encoding(tup[0]))
            b.extend(vbyte_encoding(tup[1]))
        b.extend(b'\x00')
        return b
    except:
        print(line)
        raise Exception

#decode an entry in file and then stop reading
def decode(file):
    n = 0
    numbers = []
    is_did=True
    is_freq=False
    start_seen=False
    while True:
        byte = unpack('%dB' % 1, file.read(1))[0]
        if byte == 0 and start_seen:
            break
        if byte < 128:
            n = 128 * n + byte
            start_seen=False
        else:
            n = 128 * n + (byte - 128)
            if is_did:
                did=n
                is_did=False
                is_freq=True
            elif is_freq:
                new_tup=(did,n)
                is_did=True
                is_freq=False
                numbers.append(new_tup)
                start_seen=True
            n = 0
    return numbers

#reads one entry of the inverted index
def read_encoded_line(file):
    next_byte=file.read(1)
    if not next_byte:
        return None,None
    #decode term
    term=bytearray()
    while next_byte != b'\x2c':
        term.extend(next_byte)
        next_byte=file.read(1)
    term = term.decode('utf-8')

    #decode document frequencies and return
    return term,decode(file)
#end of encoding related functions

#disk mergesort related functions below

#fills queue with lines from file
#returns 1 if successfull and 0 if file ended
def fill(queue,file):
    for _ in range(BUFFER_LEN):
        new_line = read_encoded_line(file)
        queue.appendleft(new_line)
        if new_line[0] is None:
            return False
    return True


#convert deltas to absolutes in inverted_list
def remove_deltas(inv_list):
    new_list=[]
    previous_did=0
    for tup in inv_list:
        tup = (tup[0]+previous_did,tup[1])
        previous_did=tup[0]
        new_list.append(tup)
    return new_list
    
#save line to buffer, save buffer to file if buffer is at full capacity
def save(term=None,l=None,file=None,buffer=[],done=False):
    #save to new inverted index if buffer is at max capacity, else, append to buffer
    if len(buffer) > BUFFER_LEN or done:
        while len(buffer) > 0:
            file.write(buffer.pop())
    if not done:
        buffer.appendleft(encode(term,l))

#merge two inverted lists of the same term
def merge(line1,line2):
    new_line=[]
    #remove deltas from inverted lists
    il1=remove_deltas(line1)
    il2=remove_deltas(line2)
    
    i=0
    j=0
    
    previous_did=0
    while i < len(il1) and j < len(il2):
        if il1[i][0] <= il2[j][0]:
            new_line.append((il1[i][0]-previous_did,il1[i][1]))
            previous_did=il1[i][0]
            i += 1
        else:
            new_line.append((il2[j][0]-previous_did,il2[j][1]))
            previous_did=il2[j][0]
            j += 1
 
    # Copy the remaining elements of l1
    while i < len(il1):
        new_line.append((il1[i][0]-previous_did,il1[i][1]))
        previous_did=il1[i][0]
        i += 1
 
    # Copy the remaining elements of l2
    while j < len(il2):
        new_line.append((il2[j][0]-previous_did,il2[j][1]))
        previous_did=il2[j][0]
        j += 1
    
    return new_line

#performs disk mergesort between file1 and file2, saves to out_file
def disk_mergesort(file1,file2,out_file_name):
    out_file = open(out_file_name,'wb')
    try:
        with open(file1,'rb') as f1:
            with open(file2,'rb') as f2:
                lines1,lines2 = deque(),deque()
                line1,line2=None,None
                save_buffer = deque()
                while True:
                    try:
                        #read a big chunk of lines
                        if len(lines1) == 0:
                            fill(lines1,f1)
                        if len(lines2) == 0:
                            fill(lines2,f2)
                        if not line1:
                            term1,line1 = lines1.pop()
                        if not line2:
                            term2,line2 = lines2.pop()

                        #check if a file has ended
                        if not line1 or not line2:
                            if not line1:
                                #read the rest of the file
                                while fill(lines2,f2):
                                    pass
                                #save to new file
                                while line2:
                                    save(term2,line2,out_file,save_buffer)
                                    term2,line2=lines2.pop()
                            if not line2:
                                #read the rest of the file
                                while fill(lines1,f1):
                                    pass
                                #save to new file
                                while line1:
                                    save(term1,line1,out_file,save_buffer)
                                    term1,line1=lines1.pop()
                            break

                        #try to merge
                        #term from line1 comes earlier alphabetically
                        if term1 < term2:
                            save(term1,line1,out_file,save_buffer)
                            line1=None
                        #term from line1 comes later alphabetically
                        elif term1 > term2:
                            save(term2,line2,out_file,save_buffer)
                            line2=None
                        #they are the same term
                        else:
                            new_line=merge(line1,line2)
                            save(term1,new_line,out_file,save_buffer)
                            line1,line2=None,None
                    except Exception as e:
                        raise(e)
        #save what is left at the buffer
        save(buffer=save_buffer,file=out_file,done=True)
        out_file.close()
        os.remove(file1)
        os.remove(file2)
        os.rename(out_file_name,file1)
    except Exception as e:
        out_file.close()
        raise e
#end disk mergesort functions

#class responsible for parsing
class Parser:
    def __init__(self):
        self.stem = SnowballStemmer("english").stem
        self.tokenize = word_tokenize
        #nltk.download('stopwords')
        self.stopw = set(stopwords.words('english'))
        self.remove_digits = str.maketrans('', '', string.digits)
        self.remove_punctuation = str.maketrans(dict.fromkeys(string.punctuation, " "))
    def parse(self,sentence):
        sentence = sentence.translate(self.remove_digits)
        sentence = sentence.translate(self.remove_punctuation)
        #leave only ascii characters
        sentence =''.join([char for char in sentence if ord(char) < 128])
        tokens = self.tokenize(sentence)
        tokens = [self.stem(w.lower()) for w in tokens if (w.lower() not in self.stopw)]
        return tokens
    def freq_dist(self,sentence):
        return nltk.FreqDist(self.parse(sentence)).items()

#class responsible for index creation
class IndexCreator:
    def __init__(self,n_threads,mem_limit,save_path):
        self.n_threads=n_threads
        manager = multiprocessing.Manager()
        self.file_lock = manager.Lock()
        self.variable_lock = manager.Lock()
        #self.document_index = manager.dict()
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
        print('ProcessesCreated')
        #add documents to q (l for each process)
        while True:
            next_lines = readN(corpus_generator,limit)
            if next_lines==[]:
                break
            self.queue.put(next_lines)
        print('File Done')
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
            #print('{} Got a drop of {} posts'.format(multiprocessing.current_process().name,len(corpus)))
            if len(corpus)==0:
                self.dump(index,document_index)
                del index,document_index
                gc.collect()
                break

            #check memory requirements
            memory_used=psutil.Process(os.getpid()).memory_info().rss / 1024 * 1024
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
                
                tokens = self.parser.freq_dist(document['text'])
                document_index[did] = [document['id'],len(tokens)]
                for i,(term,tf) in enumerate(tokens):
                        if term not in index.keys():
                            index[term] = []
                        index[term].append((did,tf))
            del corpus
            

            #print('{} why leaving while? {}'.format(multiprocessing.current_process().name,len(index)))
        
    def save_doc_index_and_statistics(self):
        with open(f'{self.save_path}/corpus_statistics','w') as corpus_stats:
            corpus_stats.write(str(self.total_documents.value)+','+str(float(self.avg_len.value)/self.total_documents.value)+'\n')

    #dump index to memory
    def dump(self,index,doc_index):
        with open(f'{self.save_path}/{self.flush_count.value}.txt','wb') as index_file:
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
    def parallel_merge(self):
        temps = glob.glob(f'{self.save_path}\\*.txt')
        while len(temps)>1:
            #generate tuples of different files (e.g. file1.txt,file2.txt)
            it=iter(temps)
            f_tuples = list(zip(it,it))

            #create arguments for disk mergesort parallel executions
            self.variable_lock.acquire()
            self.flush_count.value+=1
            args = [(tup[0],tup[1],tup[0]+tup[1].replace(f'{self.save_path}\\','')) for tup in f_tuples]
            self.variable_lock.release()

            with multiprocessing.Pool(self.n_threads) as pool:
                pool.starmap(func=disk_mergesort,iterable=args)
            '''
            for tup in f_tuples:
                t = Process(target=disk_mergesort, args=[tup[0],tup[1],tup[0]+tup[1].replace(f'{self.save_path}\\','')])
                threads.append(t)
                t.start()     
            for t in threads:
                t.join()
                '''
            #for tup in f_tuples:
               # disk_mergesort(tup[0],tup[1],tup[0].replace('.txt','')+tup[1].replace(f'{self.save_path}\\',''))
               # break
            temps = glob.glob(f'{self.save_path}\\*.txt')
        os.rename(temps[0],f'{self.save_path}\\invertedindex.txt')
    
    def create_lexicon(self):
        lex = {}
        #get information from inverted index
        with open(f'{self.save_path}\\invertedindex.txt','rb') as index:
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
        with open(f'{self.save_path}\\lexicon','wb') as file:
            for term in lex:
                offset,list_len = lex[term]
                self.avg_list_size+=list_len
                self.number_of_lists += 1
                file.write(term.encode('utf-8'))
                file.write(b'\x2c')
                file.write(vbyte_encoding(offset))
                file.write(vbyte_encoding(list_len))
                file.write(b'\x00')
        print(self.avg_list_size)
        self.avg_list_size = float(self.avg_list_size)/self.number_of_lists


def main(args):
    """
    Your main calls should be added here
    """
    start_time = time.time()
    index=IndexCreator(n_threads=6,mem_limit=args.memory_limit,save_path=args.index)
    corpus_generator = document_reader(args.corpus)
    index.create(corpus_generator)
    print(time.time() - start_time,"time before merge")
    index.parallel_merge()
    index.create_lexicon()
    elapsed_time = time.time()-start_time
    stats = {}
    stats['Elapsed Time'] = round(elapsed_time)
    stats['Index Size'] = round(os.path.getsize(f'{args.index}\\invertedindex.txt') / (1024 * 1024),2) #size in mb
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