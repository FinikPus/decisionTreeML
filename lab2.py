from pandas import read_csv, DataFrame, Series
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from matplotlib import mlab
train_dataset = read_csv('train.csv')
test_dataset = read_csv('test.csv')
train_dataset.info()
test_dataset.info()
dep_var = train_dataset["Survived"] #зависимая переменная
train_dataset.drop(['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
#test_dataset.drop(['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
#нормализация данных
def harmonize_data(titanic):
    #отсутствующим полям возраста присваивается медианное значение
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Age"].median()
    #пол преобразуется в числовой формат
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    #пустое место отплытия заполняется наиболее популярным S
    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    # место отплытия преобразуется в числовой формат
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    # отсутствующим полям суммы отплаты за плавание присваивается медианное значение
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    return titanic

train_harm = harmonize_data(train_dataset)
test_harm  = harmonize_data(test_dataset)
train_harm.info()
test_harm.info()


grid = {'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 'criterion':['entropy','gini'], 'random_state' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]}
gs = GridSearchCV(DecisionTreeClassifier(),grid,scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(train_harm, dep_var)
res = (
    pd.DataFrame({
        "mean_test_score": gs.cv_results_["mean_test_score"],
        "mean_fit_time": gs.cv_results_["mean_fit_time"]})
      .join(pd.io.json.json_normalize(gs.cv_results_["params"]).add_prefix("param_"))
)
print(res)
print("Best param:",gs.best_params_)
print("Best score:",gs.best_score_)
tree = DecisionTreeClassifier(max_depth=2, random_state = 21, criterion = 'entropy')

tree2 = DecisionTreeClassifier(max_depth=2, random_state = 21, criterion = 'gini')
tree.fit(train_harm,dep_var)
tree2.fit(train_harm,dep_var)
#print(tree.score(train_harm,dep_var))
#print(tree2.score(train_harm,dep_var))
xmin=0.0
xmax=18.0
dx=1.0
xlist =[]
ylist=[]
ylist2=[]

for i in range(0,8):
    xlist.append(i+1)
    tree = DecisionTreeClassifier(max_depth=i+1, random_state=1, criterion='entropy')
    tree2 = DecisionTreeClassifier(max_depth=i+1, random_state=1, criterion='gini')
    tree.fit(train_harm, dep_var)
    tree2.fit(train_harm, dep_var)
    ylist.append(tree.score(train_harm, dep_var))
    ylist2.append(tree2.score(train_harm, dep_var))
    #ylist[i] = tree.score(train_harm, dep_var)
    #ylist2[i] = tree2.score(train_harm, dep_var)
plt.plot(xlist,ylist)
plt.plot(xlist,ylist2)

plt.show()
dot_data = export_graphviz(tree, feature_names=train_harm.columns)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')

dot_data = export_graphviz(tree2, feature_names=train_harm.columns)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree2.png')

