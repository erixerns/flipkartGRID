def chunk(chunks):
    height = 480
    width = 640
    wd = width/chunks
    ht = height/chunks
    arr=[]
    for j in range(0,chunks):
        temp = []
        for i in range(0,chunks):
            temp.append((i*wd,j))
        arr.append(temp)
    return arr

