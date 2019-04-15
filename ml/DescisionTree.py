'''
Author : Shubham Sangamnerkar
Roll no : 4351

Kids : C++
Adults : Python
Legends : Sanskrit XD 
'''
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn import tree

class gV:
    df = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    column_names = []
    threshold = 1

def read_input():
    filepath = 'data.csv'
    try:
        return pd.read_csv(filepath)
    except:
        print("Error in reading the file " + filepath)
        exit(1)

def summarize_data(df):
    print("HEAD")
    print(df.head())
    print()
    print("SUMMARY")
    print(df.describe())
    print()
    print("INFO")
    print(df.info())
    
def clean_up(df):
    df.dropna()
    if 'ID' in df.columns:
        df = df.drop('ID', axis = 1)
    return df

def transform_categorical(df):
    if 'Age' in df.columns:
        df['Age'] = df['Age'].map({'< 21' : 0, '21-35' : 1, '> 35' : 2})
    if 'Income' in df.columns:
        df['Income'] = df['Income'].map({'Low' : 0, 'Medium' : 1, 'High' : 2})
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male' : 0, 'Female': 1})
    if 'Martial Status' in df:
        df['Martial Status'] = df['Martial Status'].map({'Single' : 0, 'Married' : 1})
    if 'Buys' in df:    
        df['Buys'] = df['Buys'].map({'No' : 0, 'Yes' : 1})

def split_data_train_test(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population = indices, k = test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df

def train_test_split(df, test_size, output):
    train_df, test_df = split_data_train_test(df, test_size)
    return train_df.drop(output, axis = 1), test_df.drop(output, axis = 1), train_df[output], test_df[output] 

def is_data_pure(data): 
    op_col = data[:, -1] #output column
    if len(np.unique(op_col)) == 1: #if number of classes is 1 then yes pure
        return True
    return False

def classify(data):
    #Also handles over fitting case
    op_col = data[:, -1] #output column
    op_classes, op_freqs = np.unique(op_col, return_counts = True) #get unique classes and their frequencies
    return op_classes[op_freqs.argmax()] #return class label with max freq


def potential_splits(data):
    #returns a dictionary
    #key is the column name
    #value is list of potential splits
    #eg : { 2 : [0.5, 1.5]} 2=income 
    all_possible_splits = dict()
    rows, cols = data.shape
    for i in range(cols - 1):
        all_possible_splits[i] = []
        distinct_col_vals = np.unique(data[:, i])
        for j in range(len(distinct_col_vals)):
            if j > 0:
                all_possible_splits[i].append((distinct_col_vals[j] + distinct_col_vals[j-1]) / 2.0)  
    return all_possible_splits

def split_data(data, column_split, val_split):
    col_vals = data[:, column_split]
    lower_half = data[col_vals <= val_split]
    higher_half = data[col_vals > val_split]
    return lower_half, higher_half

def find_entropy(data):
    #Determine probabilites for each class
    op_col = data[:, -1]
    outputs, freq_ops = np.unique(op_col, return_counts = True)
    probabilities = freq_ops / freq_ops.sum()
    entropy = sum(-1 * probabilities * np.log2(probabilities))
    return entropy
        
def total_entropy(lower_half, higher_half):
    p_low = len(lower_half) / (len(lower_half) + len(higher_half))
    return (p_low) * find_entropy(lower_half) + (1 - p_low) * find_entropy(higher_half)

def opt_split(data, all_possible_splits):
    lowest_entropy = 10**18
    for i in all_possible_splits:
        for j in all_possible_splits[i]:
            lower_half, higher_half = split_data(data, i, j)
            temp_entropy = total_entropy(lower_half, higher_half)
            if lowest_entropy >= temp_entropy :
                lowest_entropy = temp_entropy
                opt_col = i
                opt_val = j
    return opt_col, opt_val

def build_tree(data, column_names):
    if is_data_pure(data) or len(data) < gV.threshold :
        return classify(data)
    opt_col, opt_val = opt_split(data, potential_splits(data))
    lower_half, higher_half = split_data(data, opt_col, opt_val)
    key = "{column} <= {value}".format(column = column_names[opt_col], value = opt_val)
    cur_node = {key : []}
    cur_node[key].append(build_tree(lower_half, column_names))
    cur_node[key].append(build_tree(higher_half, column_names))
    return cur_node

def predict_class(sample, tree):
    condition = list(tree.keys())[0]
    attribute, val = condition.split("<=")
    attribute = attribute[:-1]
    val = val[1:]
    if sample[attribute] <= float(val):
        ans = tree[condition][0]
    else:
        ans = tree[condition][1]
    if isinstance(ans, dict) == False:
        return ans
    return predict_class(sample, ans)

class DescisionTreeClassifier():
    def __init__(self):
        self.tree = dict()
    def fit(self, x_train, y_train):
        df = pd.concat([x_train, y_train], axis = 1)
        self.tree = build_tree(df.values, df.columns)
    def predict(self, x_test):
        y_pred = pd.DataFrame(columns = ["Buys"])
        for index, row in x_test.iterrows():
            y_pred.loc[index] = predict_class(row, self.tree)
        return y_pred

gV.df = read_input()
gV.df = clean_up(gV.df)
transform_categorical(gV.df)
#summarize_data(gV.df)
gV.x_train, gV.x_test, gV.y_train, gV.y_test = train_test_split(gV.df, 0.3, ['Buys'])
model = DescisionTreeClassifier()
model.fit(gV.x_train, gV.y_train)
y_pred = model.predict(gV.x_test)
y_pred["Buys"] = np.int64(y_pred["Buys"])


model2 = tree.DecisionTreeClassifier(criterion = 'entropy')
model2.fit(gV.x_train, gV.y_train)
y_pred2 = model2.predict(gV.x_test)


print("Accuracy Score1 : " + str(accuracy_score(y_pred, gV.y_test)))
print("Accuracy Score2 : " + str(accuracy_score(y_pred2, gV.y_test)))
#Sample example test
data = [["< 21", "Low", "Female", "Married"]]
df = pd.DataFrame(data, columns=['Age', 'Income', 'Gender', 'Martial Status'])
transform_categorical(df)
if predict_class(df.iloc[0], model.tree):
    print("Yes")
else:
    print("No")