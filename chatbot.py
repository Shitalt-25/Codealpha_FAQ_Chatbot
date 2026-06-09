import streamlit as st
st.title("🤖 SMART FAQ CHATBOT")
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load FAQ Dataset
# ----------------------------
data = pd.read_csv("faq.csv")

questions = data["question"].tolist()
answers = data["answer"].tolist()

# ----------------------------
# Function to Find Best Answer
# ----------------------------
def get_answer(user_question):

    all_questions = questions + [user_question]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(all_questions)

    similarity_scores = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    )

    best_match_index = similarity_scores.argmax()

    confidence_score = similarity_scores[0][best_match_index]

    if confidence_score < 0.30:
        return "Sorry, I could not understand your question."

    return answers[best_match_index]

# ----------------------------
# Streamlit Page
# ----------------------------
st.set_page_config(
    page_title="Smart FAQ Chatbot",
    page_icon="🤖"
)

st.title("🤖 Smart FAQ Chatbot")

st.write(
    "Ask questions related to Artificial Intelligence and Programming."
)

st.write(
    "Developed by Tarde Shital"
)

st.markdown("---")

# ----------------------------
# Session State for Chat History
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# User Input
# ----------------------------
user_query = st.text_input(
    "Enter Your Question"
)

# ----------------------------
# Ask Button
# ----------------------------
if st.button("Ask Question"):

    if user_query.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        bot_response = get_answer(
            user_query
        )

        st.session_state.chat_history.append(
            ("You", user_query)
        )

        st.session_state.chat_history.append(
            ("Bot", bot_response)
        )

# ----------------------------
# Display Chat History
# ----------------------------
st.subheader("Chat History")

for sender, message in st.session_state.chat_history:

    if sender == "You":

        st.markdown(
            f"🧑 **You:** {message}"
        )

    else:

        st.markdown(
            f"🤖 **Bot:** {message}"
        )

# ----------------------------
# FAQ Section
# ----------------------------
st.markdown("---")

with st.expander("View Available FAQs"):

    for q in questions:

        st.write("•", q)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")

st.caption(
    "AI Internship Project - Smart FAQ Chatbot Using NLP"
)