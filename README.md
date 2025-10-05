# DATATHON-TM-81

Team TM-81
Judah Valiant Primeldi

Evan Wijaya

Dominicus Joe Felix

!!instruction for files
when seeing this kind of code please use this format --> pd.read_excel('file name.xls', sheet_name='Data', header = 1) (header should be the row above the first data even if it is unlabeled)

when seeing this kind of code please use this format df = df.iloc[:, list(range(0, 9)) + list(range(10, 31)) + [45]] (this sould be every column that is used for this case column 0 to 8, 10 to 30, and 45)

df = df.drop([2126, 2127, 2128]) (use this to drop unwanted row (usually tha last few row here value is NaN))

df.columns = ['b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR', 'LB', ....] (this is to name the feature name for each column)

to get the cleaned data as excel --> run the code --> df.to_excel("cleaned_data.xlsx", index=False)

make sure to used cleaned data in the train and test file
