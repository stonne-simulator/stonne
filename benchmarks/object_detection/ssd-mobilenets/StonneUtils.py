import os 


def getTileFileFromDimensions(tiles_path, dim1, dim2):
    tile_file='tile_'+str(dim1)+'x'+str(dim2)
    return os.path.join(tiles_path, tile_file)

def getTileFileFromConvDimensions(tiles_path, C, K, R, G):
    if(G > 1):  # We have created a common tile for all the layers that matches certain features
        tile_file='tile_groups'
        return os.path.join(tiles_path, tile_file)
    else:
        tile_file='tile_'+str(C)
        return os.path.join(tiles_path, tile_file) # We have one different tile for each C dimension. Note this works for any K and R 


