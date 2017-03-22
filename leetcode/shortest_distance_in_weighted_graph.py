import numpy as np
MAX_NUMBER = 100000
def min_1_N(graph_matrix,dp_list,n):


    for i in range(1,len(dp_list)):
        min_number = MAX_NUMBER
        for j in range(i):
            min_number = min(min_number,dp_list[j]+graph_matrix[i][j])
        dp_list[i] = min_number
    return dp_list[n]





if __name__ == '__main__':

    graph_matrix = [[MAX_NUMBER,10,MAX_NUMBER,9],
                    [10,MAX_NUMBER,MAX_NUMBER,8],
                    [MAX_NUMBER,MAX_NUMBER,MAX_NUMBER,MAX_NUMBER],
                    [9,18,MAX_NUMBER,1]]
    length = len(graph_matrix[0])
    dp_list = [0]*length
    n = 2
    print(min_1_N(graph_matrix,dp_list,n))
