import docx
import json
import joblib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from joblib import dump
import seaborn as sns




# Extract JSON objects from a document
def extract_json_from_word(doc_path):
    doc = docx.Document(doc_path)
    texts = [p.text for p in doc.paragraphs if p.text.strip()]
    json_objects = []
    flag = False
    teet = ''
    cc = 0
    for text in texts:
        if text == '{':
            flag = True
            teet = '{'
        elif '}' in text:
            flag = False
            # The ID 474 is a bit unusual and needs to be handled separately lol
            if '"474"' in teet and cc==1:
                continue
            if teet[-1] == ',' or teet[-1] == '}':
                teet = teet[:-1]
            if '"474"' in teet:
                #print(teet)
                cc += 1
            teet += '}'
            json_objects.append(json.loads(teet))
            #cc += 1
        elif flag:
            # Remove newline characters
            text = text.replace('\n', '')
            teet += text

    return json_objects


# Feature extraction
def extract_features(json_obj):
    """
    Choose fields for feature extraction, which are more important.
    e.g.: Title (TI), Source (SO), Authors (AU), Abstract (AB)
    """
    fields = ['TI', 'AB', 'PB', 'IN', 'DU', 'AB', 'PM', 'DJ', 'FTURL']

    texts = [json_obj.get(field, '') for field in fields]

    full_text = ' '.join(texts)
    return full_text


# 标记提取
def extract_label(json_obj):
    # This is the label field, which is used to determine whether it is an RCT
    return json_obj.get('If RCT or not') == 'Yes'

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=['Not RCT', 'RCT'], yticklabels=['Not RCT', 'RCT'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# 主函数
def main():
    doc1_path = 'output501-1000.docx'
    doc2_path = 'output101-500.docx'
    doc3_path = 'output1-100.docx'

    # Read JSON objects from Word documents
    json_data = extract_json_from_word(doc1_path) + extract_json_from_word(doc2_path)
    # json_data = extract_json_from_word(doc1_path)


    # Extract features and labels
    texts = [extract_features(obj) for obj in json_data]
    labels = [extract_label(obj) for obj in json_data]

    # Use CountVectorizer to convert text to a frequency vector
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(texts)

    # Convert labels to binary numerical values
    labels = [1 if label else 0 for label in labels]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Train the model
    clf = MLPClassifier(hidden_layer_sizes=(100, 50,), activation='identity', solver='lbfgs')  # Neural network performs best
    clf.fit(X_train, y_train)

    # Predict the test set
    y_pred = clf.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')

    # Compare results to see which predictions are wrong
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            print(json_data[i])
            print(f'预测为{y_pred[i]}')
            print(f'实际为{y_test[i]}')

    # Save the model, all that followed were simulations of others using the model
    dump(clf, 'model_mlp.joblib')
    load_clf = joblib.load('model_mlp.joblib')

    # Read a new Word document as the validation set
    # Please replace with the classification set path needed by the user :)
    new_json_data = extract_json_from_word(doc3_path)
    new_texts = [extract_features(obj) for obj in new_json_data]

    new_X = vectorizer.transform(new_texts)
    new_y_pred = load_clf.predict(new_X)

    new_real_y = [extract_label(obj) for obj in new_json_data]
    # Output the accuracy of the classification set
    print(f'Accuracy: {accuracy_score(new_real_y, new_y_pred):.2f}')
    # Compare which predictions are wrong
    for i in range(len(new_real_y)):
        if new_real_y[i] != new_y_pred[i]:
            print(new_json_data[i])
            print(f'预测为{new_y_pred[i]}')
            print(f'实际为{new_real_y[i]}')

    # Plot confusion matrix
    plot_confusion_matrix(new_real_y, new_y_pred, title='valid confusion matrix')
    plot_confusion_matrix(y_test, y_pred, title='test confusion matrix')




if __name__ == '__main__':
    main()
