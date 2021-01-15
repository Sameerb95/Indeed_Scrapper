import xgboost as xgb
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import re, csv

REGEX_PATTERN = r'(\n|,|\.)'

def label_generator(label):
    if label.lower() == 'data engineer':
        return 0
    elif label.lower() == 'data scientist':
        return 1
    else:
        return 2

def remove_space_and_punctuation(paragraph):
    if isinstance(paragraph, str):
        paragraph = re.sub(REGEX_PATTERN, ' ', paragraph)
    else: paragraph = 'NAN'
    return paragraph

def write_csv(predictions):
    labels = {
        0: 'Data Engineer',
        1: 'Data Scientist',
        2: 'Software Engineer'
    }
    with open('predictions.csv', 'w') as file:
        writer = csv.writer(file)
        for p in predictions:
            writer.writerow(labels[p])
    return

def load_csv(path, test=None):
    if not test:
        all_data = pd.read_csv(path, index_col=None, names=['Description', 'Position'])
        return all_data
    else:
        all_data = pd.read_csv(path, index_col=None, names=['Description'])
        return all_data

def stage_data(data, test=None):
    if not test:
        data['labels'] = data.apply(lambda row : label_generator(row['Position']), axis=1)
        data = data.drop('Position', 1)
    data['Description'] = data.apply(lambda row: remove_space_and_punctuation(row['Description']), axis=1)
    return data

def get_model(train_data):
    lbl_enc = preprocessing.LabelEncoder()
    y_train = lbl_enc.fit_transform(train_data.labels)
    x_train = train_data.Description

    ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 2), stop_words = 'english')
    ctv.fit(list(x_train))

    xtrain_ctv =  ctv.transform(x_train)

    clf = xgb.XGBClassifier(max_depth=12,
                        subsample=0.33,
                        objective='binary:logistic',
                        n_estimators=300,
                        learning_rate = 0.01)
    # parameters = {
    #             'min_child_weight': [1, 5, 10],
    #             'gamma': [0.5, 1, 1.5, 2, 5],
    #             'subsample': [0.6, 0.8, 1.0],
    #             'colsample_bytree': [0.6, 0.8, 1.0],
    #             'max_depth': [3, 4, 5]
    #           }
        
    # clf = GridSearchCV(clf, parameters, n_jobs=5,
    #                scoring='roc_auc',
    #                verbose=2, refit=True)

    clf.fit(xtrain_ctv.tocsc(), y_train)

    return clf, ctv

def predict(model, cv, test_csv_path):
    test_data = load_csv(test_csv_path, True)
    test_data = stage_data(test_data, True)
    x_test = cv.fit(list(test_data['Description']))
    x_test = cv.transform(x_test)
    return model.predict(x_test)

def run(train_data_path, test_data_path):
    train_data = load_csv(train_data_path)
    train_staged = stage_data(train_data)
    clf, cv = get_model(train_staged)
    predictions = predict(clf, cv, test_data_path)
    write_csv(predictions)


if __name__ == "__main__":
    run('combined - Copy.csv', 'Test.csv')
'''
    run('data/combined.csv', TESTING-FILE-PATH)
'''

