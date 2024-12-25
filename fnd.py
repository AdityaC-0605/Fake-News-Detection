import streamlit as st
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pickle
import seaborn as sns
import hashlib
import matplotlib.pyplot as plt

# Function to plot the graph
def plot_accuracy(accuracy_data):
    # Create a mapping from hash to model name
    hash_to_model = {
        hash_model_name('Logistic Regression'): 'Logistic Regression',
        hash_model_name('Decision Tree'): 'Decision Tree',
        hash_model_name('Gradient Boosting'): 'Gradient Boosting',
        hash_model_name('Random Forest'): 'Random Forest'
    }

    models = [hash_to_model[hash_key] for hash_key in accuracy_data.keys()]
    accuracy = list(accuracy_data.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=models, y=accuracy, ax=ax)
    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+', '',text)  # Removed b'' as it's not needed here
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text

def hash_model_name(model_name):
    return hashlib.sha256(model_name.encode()).hexdigest()

# Accuracy values (hashed model names as keys)
accuracy_data = {
    hash_model_name('Logistic Regression'): 88.37,
    hash_model_name('Decision Tree'): 75.22,
    hash_model_name('Gradient Boosting'): 80.81,
    hash_model_name('Random Forest'): 91.48
}

# Function to get output label
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Function to predict on new input
def manual_testing(news):
    # Load the models and vectorizer
    LR = pickle.load(open('C:/Users/Admin/Desktop/Ml/Fake-News-Detection/lr_model.pkl', 'rb'))
    DT = pickle.load(open('C:/Users/Admin/Desktop/Ml/Fake-News-Detection/dt_model.pkl', 'rb'))
    GB = pickle.load(open('C:/Users/Admin/Desktop/Ml/Fake-News-Detection/gb_model.pkl', 'rb'))
    RF = pickle.load(open('C:/Users/Admin/Desktop/Ml/Fake-News-Detection/rf_model.pkl', 'rb'))
    vectorization = pickle.load(open('C:/Users/Admin/Desktop/Ml/Fake-News-Detection/vectorizer.pkl', 'rb'))

    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    # Return predictions as a dictionary
    return {
        "LR Prediction": output_lable(pred_LR[0]),
        "DT Prediction": output_lable(pred_DT[0]),
        "GBC Prediction": output_lable(pred_GB[0]),
        "RFC Prediction": output_lable(pred_RF[0])
    }

# Streamlit app
def main():
    st.title("Fake News Detection")
    news = st.text_area("Enter the news text here:")
    if st.button("Predict"):
        if news:
            result = manual_testing(news)
            
            # Display results using st.write
            st.write("Predictions:")
            for model, prediction in result.items():
                st.write(f"{model}: {prediction}")
            plot_accuracy(accuracy_data)
            
        else:
            st.warning("Please enter some text.")

if __name__ == '__main__':
    main()