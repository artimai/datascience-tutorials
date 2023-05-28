"""
sources:
Harrison, Effective Pandas (EP)
"""
import astropy.timeseries
import numpy as np
import pandas as pd


sb: pd.Series = pd.Series(
    data=list(np.random.random(10)) + [np.NaN],
    dtype='float64',
    name='col0'  # column name
)
sb.name = 'col0'  # column name
sb.index.name = 'inxs'

dfb: pd.DataFrame = pd.DataFrame(
    data=list(np.random.random_sample((10, 10))) + [[np.NaN] * 10],
    columns=[f'col{n}' for n in range(10)],
    dtype='float64',
)
dfb.index.name = 'inxs'
dfb.columns.name = 'cols'


def contents():
    """
    EP, Ch4: indexing, info, dtypes
        pd.Series()
        s.index
        s.name
        s[boolean_array]
        s.astype()
        s.cat.ordered
        s.cat.reorder_categories()
        s.size
        s.count()

    EP, Ch6: operators, dunders
        s1.add(s2)
        ..sub, mul, div, pow, eq, gt, ...
        np.invert(s1)
        np.logical_and(s1, s2)
        np.logical_or(s1, s2)

    EP, Ch7: aggregation
        s.size
        s.count()
        s.nunique()
        s.is_unique
        s.is_monotonic_increasing
        s.min(), ...
        s.agg()

    EP, Ch8: conversion
        s.convert_dtypes()
        s.astype()
        pd.CategoricalDtype()
        s.to_numpy()
        s.to_list()
        s.to_frame()
        pd.to_datetime()  -> see dates

    EP, Ch9: manipulation
        s.apply,
        s.where, s.mask, np.select, s.replace
        s.isna, s.fillna, s.ffill, s.bfill, s.interpolate
        s.unique, s.duplicated, s.drop_duplicates
        s,sort_values, s.sort_index
        s.rank
        pd.cut, pd.qcut

    """


class Series:

    # Ch4

    @staticmethod
    def index():
        """
        not mutable
        types: RangeIndex, Index(['a', 'b', 'c']), DatetimeIndex, ...
        """

        # series as dict
        series = {
            'index': ['a', 'b', 'c'],
            'data': ['value1', 'value2', 'value3'],
            'name': 'example'
        }

        # indexing: series['b']
        index_value = 'b'
        index_location = series['index'].index(index_value)  # 1
        series_value = series['data'][index_location]        # 'value2'

        # object, names
        s = pd.Series([1, 2, 3], name='col0', index=pd.Index([1, 2, 3], name='inxs'))
        print(s.index)         # Index, RangeIndex, ...
        print(s.index.name)    # name of index
        print(s.name)          # name of column

        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        print(s.index)

    @staticmethod
    def indexing():

        # index value
        s = pd.Series(list('abc'), index=[0, 1, 2])
        print(s[1])

        s = pd.Series(list('abc'), index=list('xyz'))
        print(s['y'])

        # boolean
        mask = s > 'a'  # by column value
        print(s[mask])

        mask = s.index > 'x'  # by index value
        print(s[mask])

    @staticmethod
    def shape():

        s = pd.Series([1, np.NaN, None])
        print(s)
        print(s.size)
        print(s.count())  # exclude nans

    @staticmethod
    def dtypes():

        # Ch8: automatic conversion
        s = pd.Series([1, 2, 3, np.NaN])
        s.convert_dtypes()

        s = pd.Series(['a', 'b', None])
        s.convert_dtypes()

        # manual conversion
        """
        str 'str'
        int 'int64'
        'Int64'
        float 'float64'
        'category'
        """

        # numeric
        s = pd.Series([1, 2, 3])
        print(s.dtypes)
        s = s.astype(dtype='float64', errors='raise')  # errors = 'ignore'
        s = s.astype(dtype='float32')
        s = s.astype(dtype='float16')
        s = s.astype(dtype='int64')     # does not accept np.NaN
        s = s.astype(dtype='Int64')     # accepts np.NaN
        s = s.astype(dtype='int32')
        s = s.astype(dtype='int16')
        s = s.astype(dtype='int16')

        # ranges
        np.iinfo('int64')
        np.iinfo('int32')
        np.iinfo('int16')
        np.iinfo('int8')
        np.iinfo('uint8')
        np.finfo('float64')
        np.finfo('float32')
        np.finfo('float16')

        # int64, Int64
        pd.Series([1, None], dtype='int64')  # int64 does not support NA
        pd.Series([1, None], dtype='Int64')  # Pandas Int64 support NA

        # string
        s = pd.Series(['a', 'b', None])
        s.astype(str)       # Python
        s.astype('str')     # Python str, None is None
        s.astype('string')  # Pandas string, accepts NA

        # category
        ''' you can use .str, .dt with category '''
        s = pd.Series(list('dddaaaabbbcc'))  # dtype = 'O'
        s = s.astype('category')  # ordered automatically
        s.str.upper()

        s = pd.Series([5, 6, 1, 1, 1, 2, 2, 3])
        s = s.astype('category')  # ordered automatically

        # ordered category
        ordered_cat = pd.api.types.CategoricalDtype(categories=['c', 'b', 'a'], ordered=True)
        ordered_cat = pd.CategoricalDtype(categories=['c', 'b', 'a'], ordered=True)
        s = s.astype(ordered_cat)  # c < b < a
        print(s[s > 'b'])
        s = s.cat.reorder_categories(['a', 'b', 'c'], ordered=True)
        print(s[s > 'b'])

        # dates
        ''' see dates: pd.to_datetime '''

        # memory usage
        s = pd.Series([1, 2, 3, np.NaN])
        print(s.nbytes)
        s.memory_usage()
        s.memory_usage(deep=True)

        s = pd.Series([[1, 2, 3, np.NaN], [1, 2, 3, np.NaN]])
        print(s.nbytes)
        s.memory_usage()
        s.memory_usage(deep=True)

    # Ch6

    @staticmethod
    def dunders():
        """
        math operations: pandas will align index
            make sure you have:
            - indexes are unique
            - indexes are common to both series
        using methods allows for parametrization (e.g. fill if NaN)
        """

        s1 = pd.Series(list('abc'))
        s2 = pd.Series(list('def'))
        s1 + s2  # __add__

        # index alignment
        s1 = pd.Series([1, 2, 3], index=[1, 2, 3])
        s2 = pd.Series([4, 6, 8], index=[2, 2, 4])
        s1 + s2  # duplicates, nan

        # broadcasting
        s1 + 5

        # method, chaining
        s1.add(s2, fill_value=0)

        (s1
         .add(s2, fill_value=0)
         .div(2))

        # numpy methods
        s1 = pd.Series([True, True, False, False])
        s2 = pd.Series([True, False, True, False])
        np.invert(s1)
        np.logical_and(s1, s2)
        np.logical_or(s1, s2)

    # Ch7

    @staticmethod
    def aggregate():

        # boolean
        s = pd.Series([1, 3, 2, 1, 3, 3])
        print(s.is_unique)
        print(s.is_monotonic_increasing)
        print(s.is_monotonic_decreasing)

        # statistics
        print(s.size)
        print(s.count())
        print(s.sum())
        print(s.min())
        print(s.max())
        print(s.mean())
        print(s.median())
        print(s.quantile(q=[0.1, 0.5, 0.9]))
        print(s.std())
        print(s.var())

        # count and pct of attribute
        s.gt(1).sum()               # count of numbers greater than 1
        s.gt(1).mean()     # percentage of numbers greater than 1
        (s > 1).mul(100).mean()

        # .agg

        # built-in function
        s = pd.Series([1, 3, 2, 1, 3, 3])
        s.agg('mean')                   # array
        s.agg(np.mean)
        s.agg(np.sum)

        s.agg('any')
        s.agg('all')
        s.agg('size')
        s.agg('count')
        s.agg('sum')
        s.agg('cumsum')
        s.agg('prod')
        s.agg('cumprod')
        s.agg('dtypes')
        s.agg('empty')
        s.agg('hasnans')
        s.agg('nunique')
        s.agg('idxmax')
        s.agg('idxmin')
        s.agg('min')
        s.agg('max')
        s.agg('mean')
        s.agg('median')
        s.agg('quantile', q=[0.1, 0.9])
        s.agg('sem')
        s.agg('std')
        s.agg('var')
        s.agg('corr', other=s)
        s.agg('autocorr')
        s.agg('cov', other=s)
        s.agg('skew')
        s.agg('kurt')

        # builtin functions
        s.any()
        s.all()
        print(s.size)
        s.count()
        s.nunique()
        s.sum()
        s.cumsum()
        s.prod()
        s.cumprod()
        s.autocorr(lag=2)
        s.corr(other=s, method='pearson')
        s.min()
        s.cummin()
        s.max()
        s.cummax()
        s.mean()
        s.median()
        s.quantile(q=[0.1, 0.9])
        s.sem(ddof=1)
        s.std(ddof=1)
        s.var(ddof=1)
        s.cov(other=s, ddof=1)
        s.skew()
        s.kurt()

        # mad
        (s - s.mean()).abs().mean()

        # own function
        s = pd.Series([1, 3, 2, 1, 3, 3])
        s.agg(lambda x: x.sum())        # array
        s.agg(lambda x: np.sum(x))      # scalar !!!

        s = pd.Series(['Abc', 'abCdef'])
        s.agg(lambda x: x.shape[0])     # array
        s.agg(lambda x: x.size)         # array
        s.agg(lambda x: type(x))        # !!! scalar
        s.agg(lambda x: len(x))         # !!! scalar: compare to ints below
        s.agg(lambda x: x == 'Abc')     # unknown
        s.agg(lambda x: x[0] == 'Abc')  # array
        s.agg('any')                    # array
        s.agg(lambda x: x.any())        # array
        s.agg(lambda x: np.any(x))      # !!! scalar

        s = pd.Series([1, 3, 2, 1, 3, 3])
        s.agg(lambda x: x.shape[0])     # array
        s.agg(lambda x: x.size)         # array
        s.agg(lambda x: type(x))        # !!! scalar
        s.agg(lambda x: len(x))         # !!! array: compare to strings above
        s.agg(lambda x: x == 3)         # unknown
        s.agg(lambda x: x > 1)          # unknown
        s.agg(lambda x: x[0] == 3)      # array
        s.agg('any')                    # array
        s.agg(lambda x: x.any())        # array
        s.agg(lambda x: np.any(x))      # !!! scalar

        def scalar_if_equal(x):
            if x == 'Abc':
                return 'found'
            else:
                return 'not_found'

        def scalar_as_array(x):
            return np.array([x] * 2)

        def array_a_idx(a):
            return a[0]

        def array_a_sum(a):
            return a.sum()

        def array_as_array(a):
            return np.concatenate([a, a])

        def any_np_sum_a(a):
            return np.sum(a)

        def any_len_x(x):
            return len(x)

        s = pd.Series(['Abc', 'abCdef'])
        s.agg(scalar_if_equal)      # scalar
        s.agg(scalar_as_array)      # scalar
        s.agg(array_a_idx)          # !!! scalar
        s.agg(array_a_sum)          # array
        s.agg(array_as_array)       # array
        s.agg(any_np_sum_a)         # !!! array
        s.agg(any_len_x)            # !!! scalar

        s = pd.Series([1, 3, 2, 1, 3, 3])
        s.agg(scalar_if_equal)      # scalar
        s.agg(scalar_as_array)      # scalar
        s.agg(array_a_idx)          # array
        s.agg(array_a_sum)          # array
        s.agg(array_as_array)       # array
        s.agg(any_np_sum_a)         # !!! scalar
        s.agg(any_len_x)            # !!! array

        # function with parameters
        def func(a, op=None):
            if op == 'square':
                return a ** 2
            if op == 'mul2':
                return a * 2
            else:
                return a

        s = pd.Series([1, 3, 2, 1, 3, 3])
        s.agg(func, op='square')
        s.agg(func, op='mul2')
        s.agg(func, op=None)

        # multiple functions
        s = pd.Series([1, 3, 2, 1, 3, 3])
        s.agg(['mean', np.std])
        s.agg({'avg_': 'mean', 'stddev_': np.std})

        # -----------------------

    # Ch8

    @staticmethod
    def convert():

        s = pd.Series([1, 2, 3])

        s.to_numpy()  # same as s.values - > DO NOT USE!
        s.to_list()
        s.to_string()

        s.to_frame()              # default col name = 0
        s.to_frame(name='name')   # name: col name
        pd.DataFrame(s, dtype='float32')

    # Ch9

    def transform(self):

        # apply
        s = pd.Series([1, 2, 3, 1, 2, 5])
        s.apply(lambda x: x > 1)
        s.gt(1)

        def func(x: str, lower=False) -> str:
            if x == 'a':
                result = 'AA'
            elif x == 'b':
                result = 'BB'
            else:
                result = 'XX'
            if lower:
                result = result.lower()
            return result

        s = pd.Series(['a', 'b', 'c'])
        s.apply(func)
        s.apply(func, lower=True)

        # apply vs agg, transform, map
        s.apply(func)
        s.agg(func)
        s.transform(func)
        s.map(func)

        # if else
        """
        mask(cond, if_true_replace_with)
        where(cond, if_true_replace_with)
        np.select(
            ['bool_array1', 'bool_array2'], 
            ['replace_value1', 'replace_value2'], 
            'default_int')
        """
        s = pd.Series(['a', 'b', 'c'])
        s.mask(s == 'a', other='X')   # if cond then replace else keep
        s.where(s == 'a', other='X')  # if cond then keep else replace
        np.select(
            [(s == 'a'), (s == 'b')],
            ['X', 'Y'],
            'Z'
        )

        # replace
        s = pd.Series(['a', 'b', 'c', 'abbbbc'])
        s.replace(to_replace='a', value='X')
        s.replace(to_replace={'b': 'Y'})
        s.replace(to_replace=['a', 'b'], value=['x', 'y'])
        s.replace(to_replace=r'b.*', value='BX', regex=True)
        s.replace(to_replace=r'a(.*)(c)', value=r'\2-a-\1', regex=True)

    @staticmethod
    def count_groups():

        # value_counts
        s = pd.Series(list('aaaabbbcc'))
        s.value_counts()                    # counts of groups
        s.value_counts(normalize=True)      # pct of groups

        s = pd.Series([1, 2, 3, 5, 5, 6, 8, 8, 8, 8])
        s.value_counts(bins=3)                          # counts of groups, binned, sort by count
        s.value_counts(bins=[0, 4, 8], ascending=True)  # counts of groups, binned, sort by x

        # value_counts vs nlargest
        s = pd.Series([1, 2, 3, 5, 5, 6, 8, 8, 8, 8])
        print(s.value_counts()[:2])     # 2 groups with the largest number of observations
        s.nlargest(2)                   # 2 largest observations

    @staticmethod
    def missing():

        s = pd.Series([1, 2, np.NaN, None, 5], dtype='float64')

        # True if NA
        s.isna()
        s.isnull()
        s.notna()
        s.notnull()

        # count NA
        s.isna().sum()
        print(s.size - s.count())

        # show na
        missing = s.isna()
        not_missing = s.notna()
        print(s.loc[missing])
        print(s.loc[not_missing])

        # drop / replace NA
        s.dropna()
        s.fillna(0)
        s.fillna(s.mean())
        s.interpolate(method='linear')
        s.ffill()
        s.bfill()

    @staticmethod
    def outliers():

        # clip: replace outliers with lower/upper bound
        s = pd.Series([-1, 0, 1, 2, 3, 100, 101])
        s.clip(lower=s.quantile(0.1), upper=s.quantile(0.9))
        s.clip(lower=0, upper=100)
        (s
         .where(s >= 0, 0)
         .where(s <= 100, 100))

    @staticmethod
    def sort():
        s = pd.Series([1, 4, 2, 5, 3], index=['a', 'b', 'c', 'd', 'e'])

        print(s.is_monotonic_increasing)
        print(s.is_monotonic_decreasing)

        s.sort_values(ascending=True)
        s.sort_index(ascending=False)

    @staticmethod
    def duplicates():
        s = pd.Series(['a', 'a', 'b', 'c', 'c'])

        # is unique
        print(s.is_unique)
        np.invert(s.duplicated()).all()

        # is duplicated
        np.invert(s.is_unique)
        s.duplicated().any()
        s.duplicated(keep='first')  # return boolean array
        s.duplicated(keep='last')   # return boolean array

        # drop duplicates
        s.drop_duplicates(keep='last')
        is_duplicated = s.duplicated(keep='last')   # return boolean array
        print(s[~is_duplicated])

    @staticmethod
    def ranks():
        s = pd.Series([30, 20, 10, 10, 10, 40, 60, 80])
        s.rank(method='first', ascending=True, pct=False)
        s.rank(method='first', ascending=True, pct=True)

        s.rank(method='min')        # 5, 4, 1, 1, 1
        s.rank(method='average')    # 5, 4, 2, 2, 2
        s.rank(method='dense')      # 3, 2, 1, 1, 1

    @staticmethod
    def bins():
        s = pd.Series([1, 2, 3, 10, 20, 30, 100, 200, 300], dtype='float64')

        # cut vs qcut
        pd.cut(s, bins=3)   # same bin width (x)
        pd.qcut(s, q=3)     # same number of elements (y)

        # define bin width
        pd.cut(s, bins=[1, 10, 50, 200])
        pd.cut(s, bins=[1, 10, 50, 300], include_lowest=True)
        pd.cut(s, bins=[1, 10, 50, 300], include_lowest=True, right=False)

        # define number of elements
        pd.qcut(s, q=[0, 0.1, 0.5, 0.9, 1])

        # set labels
        pd.cut(s, bins=3, labels=['a', 'b', 'c'])

    def create(self):

        pd.Series(
            data=np.random.random(10),  # list, array, series
            dtype='float64',
            name='col0'  # name of column
        )


# sc = Series()
# sc.aggregate()

class DataFrame:
    ...


class Cookbook:
    ...
