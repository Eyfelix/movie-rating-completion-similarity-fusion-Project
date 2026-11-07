
import numpy as np
import os
from tqdm import tqdm, trange
import argparse
import pandas as pd
from scipy import sparse

from src.compareKnn import SimilarityFusion



def getMovieFeature():
    movieDataName = './datasets/namesngenre.npy'
    movieData = np.load(movieDataName, allow_pickle=True)
    return movieData


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    trainData = table

    movieData = getMovieFeature()
    bestParams = {"k_user":500, "k_movie":900, "delta":0.1, "alpha":0.3, "lam":0.4, "k_user_cross":100, "k_movie_cross":400, "sim":"paper", "comb":"prod", "reg_u":10, "reg_i":10,"n_iter":5 }
    
    knn = SimilarityFusion(**bestParams).fit(trainData, movieData)

    table = knn.predict()

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
