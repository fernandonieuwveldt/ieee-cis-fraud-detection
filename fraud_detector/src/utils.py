import numpy
np = numpy


def reduce_mem_usage(df):
    """
    (See: function taken from kaggle user: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage)

    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.

    :param df: dataframe to be reduced in memory
    :return: reduced memory data frame

    """

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    return df


def unique_mapper(df_train, df_test, feature_name):
    feature_train = set(df_train[feature_name].fillna(999).map(int).unique())
    feature_test = set(df_test[feature_name].fillna(999).map(int).unique())
    return feature_train & feature_test


# TODO: include kaggle reference
def make_day_feature(df, offset=0.0, tname='TransactionDT'):
    """
    Creates a day of the week feature, encoded as 0-6.

    Parameters:
    -----------
    df : pd.DataFrame
        df to manipulate.
    offset : float (default=0)
        offset (in days) to shift the start/end of a day.
    tname : str
        Name of the time column in df.
    """
    # found a good offset is 0.58
    days = df[tname] / (3600 * 24)
    encoded_days = numpy.floor(days - 1 + offset) % 7
    return encoded_days


def make_hour_feature(df, tname='TransactionDT'):
    """
    Creates an hour of the day feature, encoded as 0-23.

    Parameters:
    -----------
    df : pd.DataFrame
        df to manipulate.
    tname : str
        Name of the time column in df.
    """
    hours = df[tname] / 3600.0
    encoded_hours = numpy.floor(hours) % 24.0
    return encoded_hours
