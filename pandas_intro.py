import pandas as pd
from matplotlib import pyplot as plt
print pd.__version__

california_housing_dataframe = pd.read_csv(
    "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
# print california_housing_dataframe.describe(include='all')
# print california_housing_dataframe.hist('housing_median_age')
# plt.show()

##################################
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities_df = pd.DataFrame({'City name': city_names, 'Population': population})
cities_df['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities_df['Population density'] = cities_df['Population'] / cities_df['Area square miles']
cities_df = cities_df.reindex([2, 0, 1])
print cities_df
# print cities_df.index
