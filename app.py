import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "message"]

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Train model
cv = CountVectorizer()
X = cv.fit_transform(data["message"])
y = data["label"]

model = MultinomialNB()
model.fit(X, y)

# UI
st.title("📧 Spam Message Detection")
st.write("Enter a message to check whether it is Spam or Not Spam.")

user_input = st.text_area("Enter your message")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        test = cv.transform([user_input])
        result = model.predict(test)

        if result[0] == 1:
            st.error("🚫 This is a SPAM message")
        else:
            st.success("✅ This is NOT a spam message")