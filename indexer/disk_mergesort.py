import os
from collections import deque
from encoding import encode, read_encoded_line, remove_deltas
#this file contains functions used for disk mergesort

#postings lists to be read at once or saved at once
BUFFER_LEN = 5000 

#fills queue with lines from file
#returns 1 if successfull and 0 if file ended
def fill(queue,file):
    for _ in range(BUFFER_LEN):
        new_line = read_encoded_line(file)
        queue.appendleft(new_line)
        if new_line[0] is None:
            return False
    return True


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