import pandas as pd
import numpy as np
import pickle

# def md_prediction(temp,humid,light,soil,ec):
#     data = [temp,humid,light,soil,ec]

#     x_test = np.array(data)
#     x_test = x_test.reshape((1,-1))

#     filename = "classification.sav"
#     loaded_model = pickle.load(open(filename, 'rb'))

#     result = loaded_model.predict(x_test)

#     return result

def md_prediction(temp):
    x_test = np.array(temp)
    x_test = x_test.reshape((1,-1))

    filename = "temp_estimate_humid.sav"
    loaded_model = pickle.load(open(filename, 'rb'))

    result = loaded_model.predict(x_test)
 
    return result


