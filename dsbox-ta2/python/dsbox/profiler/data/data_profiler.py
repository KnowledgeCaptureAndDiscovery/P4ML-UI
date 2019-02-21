import pandas as pd
import numpy as np

# This is a pared-down version of the main Data profiling entry point
# for faster processing

# from feature computation functions
import dsbox.datapreprocessing.profiler.feature_compute_lfh as fc_lfh
import dsbox.datapreprocessing.profiler.feature_compute_hih as fc_hih
from collections import defaultdict

class DataProfiler(object):
    """
    Converted the main function of the data profiler into a class
    """
    def __init__(self, data):
        self.result = self.profile_data(data)

    def profile_data(self, data, punctuation_outlier_weight=3,
            numerical_outlier_weight=3, token_delimiter=" ", detect_language=False, topk=10):
        """
        Main function to profile the data.
        Parameters
        ----------
        data_path: file or pandas DataFrame that needs to be profiled
        ----------
        """
        isDF = True

        result = {} # final result: dict of dict
        for column_name in data:
            col = data[column_name]

            # Get a sample value of the column
            samplevalue = None

            validindices = col.index.get_indexer(col.index[~col.isnull()])
            if len(validindices > 0):
                samplevalue = col.iloc[validindices[0]]

            # dict: map feature name to content
            each_res = defaultdict(lambda: defaultdict())

            if col.dtype.kind in np.typecodes['AllInteger']+'uMmf':
                each_res["missing"]["num_missing"] = pd.isnull(col).sum()
                each_res["missing"]["num_nonblank"] = col.count()
                each_res["special_type"]["dtype"] = str(col.dtype)
                ndistinct = col.nunique()
                each_res["distinct"]["num_distinct_values"] = ndistinct
                each_res["distinct"]["ratio_distinct_values"] = ndistinct/ float(col.size)

            if col.dtype.kind == 'b':
                each_res["special_type"]["data_type"] = 'bool'
                #fc_hih.compute_common_values(col.dropna().astype(str), each_res, topk)

            elif col.dtype.kind in np.typecodes['AllInteger']+'u':
                each_res["special_type"]["data_type"] = 'integer'
                fc_hih.compute_numerics(col, each_res)
                #fc_hih.compute_common_values(col.dropna().astype(str), each_res,topk)

            elif col.dtype.kind == 'f':
                each_res["special_type"]["data_type"] = "float"
                fc_hih.compute_numerics(col, each_res)
                #fc_hih.compute_common_values(col.dropna().astype(str), each_res,topk)

            elif col.dtype.kind == 'M':
                each_res["special_type"]["data_type"] = "datetime"

            elif col.dtype.kind == 'm':
                each_res["special_type"]["data_type"] = "timedelta"

            elif col.dtype.kind == 'O' and isinstance(samplevalue, (np.ndarray, list, tuple)):
                each_res["special_type"]["data_type"] = "list"

            else:
                if isDF:
                    if col.dtype.name == 'category':
                        each_res["special_type"]["data_type"] = 'category'
                    col = col.astype(object).fillna('').astype(str)

                # compute_missing_space Must be put as the first one because it may change the data content, see function def for details
                #fc_lfh.compute_missing_space(col, each_res)
                #fc_lfh.compute_filename(col, each_res)
                fc_lfh.compute_length_distinct(col, each_res, delimiter=token_delimiter)
                #if detect_language: fc_lfh.compute_lang(col, each_res)
                #fc_lfh.compute_punctuation(col, each_res, weight_outlier=punctuation_outlier_weight)

                fc_hih.compute_numerics(col, each_res)
                #fc_hih.compute_common_numeric_tokens(col, each_res,topk)
                #fc_hih.compute_common_alphanumeric_tokens(col, each_res, topk)
                #fc_hih.compute_common_values(col, each_res, topk)
                #fc_hih.compute_common_tokens(col, each_res, topk)
                #fc_hih.compute_numeric_density(col, each_res)
                #fc_hih.compute_contain_numeric_values(col, each_res)
                #fc_hih.compute_common_tokens_by_puncs(col, each_res, topk)

            if not each_res["numeric_stats"]: del each_res["numeric_stats"]

            result[column_name] = each_res # add this column features into final result

        return result
