def add_watermark(image):
    for x in range(4, 12):
        image[x][4] = [50, 200, 3]
        image[x][5] = [50, 200, 3]
        image[x][11] = [50, 200, 3]
        image[x][12] = [50, 200, 3]
        image[x][21] = [0, 0, 0]
        image[x][29] = [0, 0, 0]
        image[x][22] = [0, 0, 0]
        image[x][30] = [0, 0, 0]
    for y in range(4, 12):
        image[4][y] = [50, 200, 3]
        image[5][y] = [50, 200, 3]
        image[11][y] = [50, 200, 3]
        image[12][y] = [50, 200, 3]
        image[21][y] = [200, 50, 3]
        image[22][y] = [200, 50, 3]
        image[29][y] = [200, 50, 3]
        image[30][y] = [200, 50, 3]
    return image