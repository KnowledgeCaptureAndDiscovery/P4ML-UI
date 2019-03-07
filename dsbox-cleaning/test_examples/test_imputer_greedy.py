"""
sample program for classification problem
"""
def text2int(col):
    """
    convert column value from text to integer codes (0,1,2...)
    """
    return pd.DataFrame(col.astype('category').cat.codes,columns=[col.name])

import pandas as pd

from dsbox.datapreprocessing.cleaner import GreedyImputation

# STEP 1: get data
data_path = "../../dsbox-data/o_38/encoded/"
data_name = data_path + "trainData_encoded.csv"
label_name = data_path + "trainTargets_encoded.csv" # make sure your label target is in the second column of this file

data = pd.read_csv(data_name)
label = text2int(pd.read_csv(label_name)["Class"])

data.drop("d3mIndex",axis=1)    # drop because id, useless

# STEP 2: go to use the Imputer !
# check GreedyImputation
imputer = GreedyImputation(verbose=1)
imputer.set_training_data(inputs=data, outputs=label)	# unsupervised
imputer.fit(timeout=10)	# give 10 seconds to fit
print (imputer.get_call_metadata())	# to see wether fit worked
result = imputer.produce(inputs=data, timeout=0.01)
print (imputer.get_call_metadata())	# to see wether produce worked

