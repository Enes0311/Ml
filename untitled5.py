import numpy as np
import pandas as pd
import sklearn.linear_model
import matplotlib.pyplot as plt

# Define the function of country_stats dataframe
def prepare_country_stats(oecd_bli, gdp_per_capita):
  
    # For example, you can merge the two dataframes based on a common column, clean missing values, etc.
    #  the resulting dataframe has columns like "GDP per capita" and "Life satisfaction"

    # For demonstration purposes, let's assume you have already prepared the country_stats dataframe
    # and it contains columns "GDP per capita" and "Life satisfaction"
    country_stats = pd.DataFrame({
        "Subject Descriptor": gdp_per_capita["Subject Descriptor"],
        "Indicator": oecd_bli["Indicator"]
    })

    return country_stats

# import the data from excel files ......
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

#by using the defined function we can prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

# Rest of your code remains unchanged...

X = np.c_[country_stats["Subject Descriptor"]]
y = np.c_[country_stats["Indicator"]]
country_stats.plot(kind='scatter', x="Subject Descriptor", y='Indicator')
plt.show()

model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new))  # outputs [[ 5.96242338]]