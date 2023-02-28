import pandas as pd
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets

df_test = pd.read_csv('processed_test.csv', sep=',')
df_train = pd.read_csv('processed_train.csv', sep=',')


# Drop rows with missing values (these were punctuations)
df_train.dropna(subset=['WORD', 'XPOS'], inplace=True)

# Pick logistic regression model

model = LogisticRegression(max_iter=100)

# Pick features to use for classification
features = ['WORD','LEMMA','BASIC DEP', 'ENHANCED DEPS','PREDICATE', 'XPOS', 'MORPH', 'ENTITY', 'WORDPOS']


# Create sets to train model

def select_features(df_train, df_test, selected_features):
    train_features = df_train[selected_features].to_dict('records')
    test_features = df_test[selected_features].to_dict('records')
    return train_features, test_features


def create_trainingset(train, test, features):
    train_features, test_features = select_features(train, test, features)
    vec = DictVectorizer()
    X_train, Y_train = vec.fit_transform(train_features), train['CATEGORY']
    X_test, Y_test = vec.transform(test_features), test['CATEGORY']

    return X_train, Y_train, X_test, Y_test


# Create training and test sets

X_train, Y_train, X_test, Y_test = create_trainingset(df_train, df_test, features)

# Train the model

print(f'Training model...')

model.fit(X_train, Y_train)

# Predict with trained model

preds = model.predict(X_test)

# Evaluate the model and print the results

report = pd.DataFrame(
    classification_report(y_true=Y_test, y_pred=preds, output_dict=True, zero_division=0)).transpose()
print(report)
confusion_matrix = confusion_matrix(Y_test, preds)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                    display_labels=['ARG0', 'ARG1', 'ARG1-DSP', 'ARG2', 'ARG3', 'ARG4', 'ARGM-ADJ',
                                                    'ARGM-ADV', 'ARGM-CAU', 'ARGM-DIR','ARGM-COM ','ARGM-DIS', 'ARGM-EXT',
                                                    'ARGM-GOL', 'ARGM-LOC', 'ARGM-MNR', 'ARGM-MOD',
                                                    'ARGM-NEG', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR',
                                                    'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG1-DSP', 'C-V', '0', 'R-ARG0',
                                                    'R-ARG1', 'V']).plot(cmap='Blues', xticks_rotation='vertical')

plt.show()