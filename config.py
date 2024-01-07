model_features = [
    'Close',
    'Rolling_Mean',
    'Rolling_Std',
    'Price_RoC',
    'EMA',
    'Short_EMA',
    'Long_EMA',
    'MACD',
    'Signal_Line',
    'Price_Delta',
    'Gain',
    'Loss',
    'RS',
    'RSI',
    'Upper_Band',
    'Lower_Band',
    'Z_Score',
    'Cumulative_Returns',
    'Log_Returns',
    'Moving_Average_5',
    'Moving_Average_10',
    'Moving_Average_15',
    'Price_RoC_30',
    'Price_RoC_45',
    'Price_RoC_60',
    'Price_RoC_5',
    'Price_RoC_10',
    'Price_RoC_15',
    'Std_Dev_RoC_5',
    'Std_Dev_RoC_10',
    'Expanding_Mean',
    'Price_RoC_3',
    'Price_RoC_7',
    'Price_RoC_12',
    'Moving_Average_3',
    'Square_Root_Close',
    'Log_Close',
    'Close_Squared',
    'Close_Cubed',
    'Close_Inverse',
    'Sin_Close',
    'Cos_Close',
    'Tan_Close'
]
model_target = 'Close'
constants = ['Date']

correlation_bar = 0.7
look_back = 5
epochs = 100
train_test_split = 0.8

days_to_predict = 15
start_date = "2021-12-01"
end_date = "2023-12-15"
clearPrevious = True
to_predict = False

is_plot = True
csv_file = "data/stocks.csv"

ignored_features = ['Expanding_Std']
