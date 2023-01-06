import numpy as np

def split3fold(data, blackboxtrain_size, test_size, random_state=0):
    """Split pandas dataframe into 3 subsets
    """
    np.random.seed(random_state)
    test_size = 1-test_size
    blackboxtrain, train, test = np.split(data.sample(frac=1, random_state=random_state),
                                          [int(blackboxtrain_size*len(data)), int(test_size*len(data))])
    return [blackboxtrain, train, test]