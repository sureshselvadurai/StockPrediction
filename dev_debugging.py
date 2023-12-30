list1 = ['Unnamed: 0', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
         'Dividends', 'Stock Splits', 'DayOfWeek', 'Rolling_Mean', 'Rolling_Std',
         'Price_RoC', 'EMA', 'Short_EMA', 'Long_EMA', 'MACD', 'Signal_Line',
         'Price_Delta', 'Gain', 'Loss', 'RS', 'RSI']

list2 = ['Close', 'DayOfWeek', 'Rolling_Std', 'EMA', 'Long_EMA', 'MACD', 'Gain', 'Loss']

# Convert lists to sets
set1 = set(list1)
set2 = set(list2)

# Find the difference
difference = set2 - set1

print("Element in the second list not present in the first list:", difference)

