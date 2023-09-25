# Weather Prediction Python App


## Table of Contents
* [Introduction](#introduction)
* [Weather Data Review](#weather-data-review)
* [Code Exploration](#code-exploration)
## Introduction
Every person looks at the weather or makes note of the weather before going outside, what if we could predict the weather without having to look at the news or other such sources.

This project aims to examine historical weather data from JFK International Airport in NY, USA to predict the weather of tomorrow.
## Weather Data Review
The dataset used for this project was sourced from [NOAA](https://www.ncdc.noaa.gov/cdo-web/), which contains valuable information regarding the weather. [weather.csv](https://github.com/jidafan/weather-predict/blob/main/weather.csv), provides information on various variables that describe the weather affecting JFK International Airport from January 1st, 1970 to August 31, 2023.


| Variable      | Description           | 
| ------------- |:---------------------| 
| `STATION`     | Meteorological station ID     |
| `NAME`     | Name of Airport           |   
| `DATE` | Date the weather was recorded                                          |
| `ACMH`  | Average cloudiness from midnight to midnight from manual observations (percentage)                                       |
| `ACSH`  | Average cloudiness from sunrise to sunset from manual observations (percentage)                                   |
| `AWND`  | Average daily wind speed (mph)                                        |
| `FMTM`  | Time of fastest mile or fastest 1-minute wind (hours and minutes,i.e., HHMM)                                      |
| `PGTM`  | Peak gust time (hours and minutes, i.e., HHMM)                                   |
| `PRCP`  | Precipitation in inches                                       |
| `SNOW`  | Snowfall in inches                                         |
| `TAVG`  | Average Temperature in Fahrenheit                                       |
| `TMAX`  | Maximum Temperature in Fahrenheit                                         |
| `TMIN`  | Minimum Temperature in Fahrenheit                                |
| `TSUN`  | Daily Total Sunshine (minutes)                                         |
| `WDF1`  | Direction of fastest 1-minute wind (degrees)                                        |
| `WDF2`  | Direction of fastest 2-minute wind (degrees)                                         |
| `WDF5`  | Direction of fastest 5-minute wind (degrees)                                   |
| `WDFG`  | Direction of peak wind gust (degrees)                                         |
| `WDFM`  | Fastest mile wind direction (degrees)                                         |
| `WESD`  | Water equivalent of snow on the ground in inches                                       |
| `WSF1`  | Fastest 1-minute wind speed (mph)                                         |
| `WSF2`  | Fastest 2-minute wind speed (mph)                                          |
| `WSF5`  | Fastest 5-minute wind speed (mph)                                           |
| `WSFG`  | Peak gust wind speed (mph)                                          |
| `WSFM`  | Fastest mile wind speed (mph)                                   |
| `WT01`  | Fog, ice fog, or freezing fog                                    |
| `WT02`  | Heavy fog or heaving freezing fog (not always distinguished from fog)                                        |
| `WT03`  | Thunder                                         |
| `WT04`  | Ice pellets, sleet, snow pellets, or small hail                                           |
| `WT05`  | Hail (may include small hail)                                         |
| `WT06`  | Glaze or rime                                       |
| `WT07`  | Dust, volcanic ash, blowing dust, blowing sand, or blowing obstruction                                      |
| `WT08`  | Smoke or haze                                          |
| `WT09`  | Blowing or drifting snow                                         |
| `WT10`  | Tornado, waterspout, or funnel cloud                                       |
| `WT11`  | High or damaging winds                                         |
| `WT12`  | Blowing spray                                        |
| `WT13`  | Mist                                        |
| `WT14`  | Drizzle                                       |
| `WT15`  | Freezing drizzle                                         |
| `WT16`  | Rain (may include freezing rain, drizzle, and freezing drizzle)                                        |
| `WT17`  | Freezing rain                                          |
| `WT18`  | Snow, snow pellets, snow grains, or ice crystals                                        |
| `WT19`  | Unknown source of precipitation                                        |
| `WT21`  | Ground fog                                           |
| `WT22`  |  Ice fog or freezing fog                                         |
| `WV01`  | In the vicinity, fog, ice fog, or freezing fog (may include heavy fog)                                          |

The file contains 46 variables and 322988 observations

## Code Exploration
**To view the full python code for this project, [click here](https://github.com/jidafan/weather-predict/blob/main/predict.py).**

**Importing Relevant Libraries and Importing Data**
```python
#Import relevant libraries
import pandas as pd
import matplotlib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
#Import data
weather = pd.read_csv("weather.csv", index_col = "DATE")
```

**Looking at the data**

```python
weather.head()
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/26d5ac01-0e8f-447a-8562-5ba579c7f5ce)

**Looking at the percentage of cells that are null values**
```python
null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]
null_pct
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/258434d9-929e-4982-9760-84979a3e36cc)

**Looking at the columns where less than 5% of the observations are null values**
```python
valid_columns = weather.columns[null_pct < .05]
valid_columns
```
**Overwriting the weather dataframe and only included the valid columns**
```python
weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()
weather
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/26299d1b-17ae-45c4-87c1-9323c88ea3ad)

**Replacing the null values with the last non empty value, and checking if they are null cells remaining**
```python
weather = weather.ffill()
weather.apply(pd.isnull).sum()
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/6442084f-71c1-4bb7-919e-431ae656548c)

**Checking the datatype for each column in the weather dataframe**
```python
weather.dtypes
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/84169282-06e7-4910-bcf6-981992a79a99)

**Looking at the index and then converting to datatime**
```python
weather.index
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/7105b5ea-7ce3-4e9a-ade8-c74e34f426e3)
```python
weather.index = pd.to_datetime(weather.index)
```

**Checking if there are any gaps in the years**
```python
weather.index.year.value_counts().sort_index()
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/c679a2cd-9ecb-41f1-a91c-932b446e311a)

**Creating a snow depth plot to see if there any gaps in the data**
```python
weather["snwd"].plot()
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/3aab9977-e2d7-43ec-aaed-a26b63462f74)

**Creating a target column for our machine learning, by using shift on the tmax column**
```python
weather["target"] = weather.shift(-1)["tmax"]
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/ef4ba4af-6ae0-4369-a096-97b66915eebb)

**Flashfilling to fill in the last value of the dataframe and checking for collinearity in the dataframe
```python
weather = weather.ffill()
#Check for collinearity
weather.corr(numeric_only=True)
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/7355b28c-4a15-4307-a3d3-5ad59fc4dcd4)

The matrix shows the correlation between the various different columns in the dataframe -1 being a strong negative correlation and 1 being a strong positive correlation

**Initializing the ridge regression model and creating a list of of predictor columns**
```python
rr = Ridge(alpha=.1)

predictors = weather.columns[~weather.columns.isin(["target","name","station"])]
```
**Creating our backtest function**
```python
def backtest(weather, model, predictors, start=3650, step = 90):
    all_predictions = []
    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]
        
        model.fit(train[predictors], train["target"])
        
        preds = model.predict(test[predictors])
        
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"],preds],axis = 1)
        
        combined.columns = ["actual", "prediction"]
        
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        
        all_predictions.append(combined)
        
    return pd.concat(all_predictions)
```
We use a backtest function as we can't use future data to predict the past. The function takes in weather dataframe, ridge regression model, list of predictors, start dictates how much data we take in before we start predicting, and step indicates we predict every 90 days.

The for loop in the function iterates from our defined start to the end of the dataframe. It creates a training set and test for our machine learning model, we convert our predictions into a dataframe and then combine them into a dataframe called combined.

![image](https://github.com/jidafan/weather-predict/assets/141703009/fd25f5dc-467c-4812-aa11-920fe9fe1f25)

**Using our error metric to assess the accuracy of our predictions**
```python
mean_absolute_error(predictions["actual"],predictions["prediction"])
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/b89ddce3-c7a3-428e-b596-a24775468491)

This means on average we were 5 degrees off from the actual temperature

**Creating functions to help improve our accuracy**
```python

def pct_diff(old, new):
    return (new - old) / old

def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather

```
Computes the rolling average 

**Rolling Horizon**
```python
rolling_horizons = [3,14]
for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)
```
Computes a 3 day rolling horizon and 14 day rolling horizon

**Removes the first 14 rows from the dataframe and fills in null values with 0**
```python
weather = weather.iloc[14:,:]
weather = weather.fillna(0)
```

**Creating function that returns the mean of all the rows together**
```python
def expand_mean(df):
    return df.expanding(1).mean()

for col in ["tmax", "tmin", "prcp"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys = False).apply(expand_mean)
    weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys = False).apply(expand_mean)

```
**Includes new predictors into predictions variable**
```python
predictors = weather.columns[~weather.columns.isin(["target","name","station"])]
predictions = backtest(weather, rr, predictors)
```

**Checks mean absolute error again**
```python
mean_absolute_error(predictions["actual"],predictions["prediction"])
```
![image](https://github.com/jidafan/weather-predict/assets/141703009/411066a3-01ed-4925-b2ee-c535f6fc1f58)

The mean absolute error has decreased from last time, which means that our accuracy has increased



