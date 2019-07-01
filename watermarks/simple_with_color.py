def add_watermark(image):
    for x in range(4,12):
        image[x][4] = [0,0,0]
        image[x][5] = [0,0,0]
        image[x][11] = [0,0,0]
        image[x][12] = [0,0,0]
        image[x][21] = [200, 50, 3]
        image[x][22] = [200, 50, 3]
        image[x][29] = [200, 50, 3]
        image[x][30] = [200, 50, 3]
    for y in range(4,12):
        image[4][y] = [0,0,0]
        image[5][y] = [0,0,0]
        image[11][y] = [0,0,0]
        image[12][y] = [0,0,0]
        image[21][y] = [50, 200, 3]
        image[22][y] = [50, 200, 3]
        image[29][y] = [50, 200, 3]
        image[30][y] = [50, 200, 3]
    return image