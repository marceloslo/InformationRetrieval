from struct import pack, unpack

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

#convert deltas to absolutes in inverted_list
def remove_deltas(inv_list):
    new_list=[]
    previous_did=0
    for tup in inv_list:
        tup = (tup[0]+previous_did,tup[1])
        previous_did=tup[0]
        new_list.append(tup)
    return new_list