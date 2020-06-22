# load dataset
df2 = pd.read_csv('data/titanic.csv')
df2.info()

# compute mean values for each class
mean_age_cls0 = df2[df2['Survived'] == 0]['Age'].mean()
mean_age_cls1 = df2[df2['Survived'] == 1]['Age'].mean()
print('mean age of not survived:', mean_age_cls0)
print('mean age of survived:', mean_age_cls1)

# replace missing values with mean
df2.loc[(df2.Age.isnull()) & (df2['Survived'] == 0), "Age"] = mean_age_cls0
df2.loc[(df2.Age.isnull()) & (df2['Survived'] == 1), "Age"] = mean_age_cls1

df2.info()
