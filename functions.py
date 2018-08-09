def numOut(layerNum, topology):
    if layerNum == (len(topology)-1):
        return 0
    else:
        return topology[layerNum+1]
