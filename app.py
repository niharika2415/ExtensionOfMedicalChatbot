import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from googletrans import Translator
from langdetect import detect

# --- Global Components ---
translator = Translator()

# --- Helper Functions ---

@st.cache_data
def load_data():
    """
    Creates a small, hard-coded multilingual subset of the MedQuAD dataset.
    This version includes translations for demonstration purposes.
    """
    st.info("Using embedded medical data...")
    data = [
        # English
        {"language": "en", "question": "What is diabetes?", "answer": "Diabetes is a chronic, metabolic disease characterized by elevated levels of blood glucose (or blood sugar), which leads over time to serious damage to the heart, blood vessels, eyes, kidneys and nerves."},
        {"language": "en", "question": "Symptoms of influenza?", "answer": "The most common symptoms of influenza are fever, cough, sore throat, and muscle aches. It is a viral infection that attacks your respiratory system."},
        {"language": "en", "question": "How to treat a headache?", "answer": "Headaches can often be treated with over-the-counter pain relievers like ibuprofen or acetaminophen. Rest and staying hydrated can also help."},
        {"language": "en", "question": "What is hypertension?", "answer": "Hypertension, also known as high blood pressure, is a serious medical condition. It can be caused by various factors and is a major risk factor for cardiovascular disease."},
        {"language": "en", "question": "What is the function of penicillin?", "answer": "Penicillin is a group of antibiotics used to treat a wide range of bacterial infections. It works by interfering with the formation of the bacteria's cell wall."},
        # Spanish
        {"language": "es", "question": "¿Qué es la diabetes?", "answer": "La diabetes es una enfermedad crónica y metabólica caracterizada por niveles elevados de glucosa en la sangre (o azúcar en la sangre), que con el tiempo provoca daños graves en el corazón, los vasos sanguíneos, los ojos, los riñones y los nervios."},
        {"language": "es", "question": "¿Cuáles son los síntomas de la gripe?", "answer": "Los síntomas más comunes de la gripe son fiebre, tos, dolor de garganta y dolores musculares. Es una infección viral que ataca el sistema respiratorio."},
        {"language": "es", "question": "¿Cómo se trata un dolor de cabeza?", "answer": "Los dolores de cabeza a menudo se pueden tratar con analgésicos de venta libre como el ibuprofeno o el paracetamol. El descanso y mantenerse hidratado también pueden ayudar."},
        {"language": "es", "question": "¿Qué es la hipertensión?", "answer": "La hipertensión, también conocida como presión arterial alta, es una afección médica grave. Puede ser causada por varios factores y es un factor de riesgo importante para las enfermedades cardiovasculares."},
        {"language": "es", "question": "¿Cuál es la función de la penicilina?", "answer": "La penicilina es un grupo de antibióticos que se utilizan para tratar una amplia gama de infecciones bacterianas. Actúa interfiriendo en la formación de la pared celular de las bacterias."},
        # German
        {"language": "de", "question": "Was ist Diabetes?", "answer": "Diabetes ist eine chronische Stoffwechselkrankheit, die durch erhöhte Blutzuckerwerte gekennzeichnet ist und im Laufe der Zeit zu schweren Schäden an Herz, Blutgefäßen, Augen, Nieren und Nerven führt."},
        {"language": "de", "question": "Symptome einer Grippe?", "answer": "Die häufigsten Symptome der Grippe sind Fieber, Husten, Halsschmerzen und Muskelschmerzen. Es handelt sich um eine Virusinfektion, die das Atmungssystem angreift."},
        {"language": "de", "question": "Wie behandelt man Kopfschmerzen?", "answer": "Kopfschmerzen können oft mit rezeptfreien Schmerzmitteln wie Ibuprofen oder Paracetamol behandelt werden. Auch Ruhe und ausreichende Flüssigkeitszufuhr können helfen."},
        {"language": "de", "question": "Was ist Bluthochdruck?", "answer": "Bluthochdruck, auch als Hypertonie bekannt, ist eine ernste Erkrankung. Sie kann durch verschiedene Faktoren verursacht werden und ist ein wichtiger Risikofaktor für Herz-Kreislauf-Erkrankungen."},
        {"language": "de", "question": "Was ist die Funktion von Penicillin?", "answer": "Penicillin ist eine Gruppe von Antibiotika, die zur Behandlung einer Vielzahl bakterieller Infektionen eingesetzt werden. Es wirkt, indem es die Bildung der Zellwand der Bakterien stört."},
        # French
        {"language": "fr", "question": "Qu'est-ce que le diabète?", "answer": "Le diabète est une maladie chronique et métabolique caractérisée par des taux élevés de glucose dans le sang (ou de sucre dans le sang), ce qui entraîne au fil du temps de graves dommages au cœur, aux vaisseaux sanguins, aux yeux, aux reins et aux nerfs."},
        {"language": "fr", "question": "Symptômes de la grippe?", "answer": "Les symptômes les plus courants de la grippe sont la fièvre, la toux, les maux de gorge et les douleurs musculaires. C'est une infection virale qui attaque votre système respiratoire."},
        {"language": "fr", "question": "Comment traiter un mal de tête?", "answer": "Les maux de tête peuvent souvent être traités avec des analgésiques en vente libre comme l'ibuprofène ou le paracétamol. Le repos et une bonne hydratation peuvent également aider."},
        {"language": "fr", "question": "Qu'est-ce que l'hypertension?", "answer": "L'hypertension, également connue sous le nom de pression artérielle élevée, est une maladie grave. Elle peut être causée par divers facteurs et constitue un facteur de risque majeur de maladies cardiovasculaires."},
        {"language": "fr", "question": "Quelle est la fonction de la pénicilline?", "answer": "La pénicilline est un groupe d'antibiotiques utilisés pour traiter un large éventail d'infections bactériennes. Elle agit en interférant avec la formation de la paroi cellulaire des bactéries."},
    ]
    df = pd.DataFrame(data)
    st.success("Data loaded successfully!")
    return df

def find_best_answer(question, df_en, vectorizer, tfidf_matrix):
    """
    Uses TF-IDF and cosine similarity on the English dataset to find the most relevant answer.
    """
    try:
        # We use a separate dataframe for the English questions to avoid
        # conflicts with questions in other languages.
        df_en.dropna(subset=['question'], inplace=True)
        query_vec = vectorizer.transform([question])
    except ValueError:
        return "I'm sorry, I couldn't find a good answer for that question. Can you please rephrase?"

    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    most_similar_index = cosine_similarities.argmax()

    if cosine_similarities[most_similar_index] > 0.1:
        return df_en.iloc[most_similar_index]['answer']
    else:
        return "I'm sorry, I couldn't find a good answer for that question. Can you please rephrase?"

def recognize_medical_entities(text):
    """
    Identifies basic medical entities in a given text using a simple dictionary lookup.
    """
    entities = []
    medical_terms = {
        'symptoms': ['fever', 'headache', 'cough', 'nausea', 'fatigue', 'dizziness'],
        'diseases': ['diabetes', 'hypertension', 'influenza', 'asthma', 'cancer'],
        'drugs': ['ibuprofen', 'acetaminophen', 'aspirin', 'penicillin', 'lipitor']
    }

    for term_type, terms in medical_terms.items():
        for term in terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
                entities.append(term)
    
    return entities

# --- Main Streamlit Application ---

def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="💊")
    st.title("👨‍⚕️ Multilingual Medical Q&A Chatbot")
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 16px;
        }
        .stTextInput>div>div>input {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("Ask a medical question in English, Spanish, German, or French.")

    # Load data
    df = load_data()

    if df.empty:
        st.error("Failed to load data. The chatbot cannot function.")
        st.stop()
    
    # Filter for English questions to build the TF-IDF matrix
    df_en = df[df['language'] == 'en'].reset_index(drop=True)
    if df_en.empty:
        st.error("No English data available for retrieval.")
        st.stop()

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_en['question'])

    # Get user input
    user_question = st.text_input("Enter your question here:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Processing your query..."):
                try:
                    # Detect the user's language
                    detected_lang = detect(user_question)
                    st.write(f"Detected language: **{detected_lang}**")

                    # Translate the question to English for retrieval
                    english_query = translator.translate(user_question, src=detected_lang, dest='en').text

                    # Find the best answer in English
                    best_answer_en = find_best_answer(english_query, df_en, vectorizer, tfidf_matrix)

                    # Translate the best answer back to the user's original language
                    final_answer = translator.translate(best_answer_en, src='en', dest=detected_lang).text
                
                    st.subheader("Answer:")
                    st.write(final_answer)

                    # Recognize and display medical entities
                    entities = recognize_medical_entities(user_question)
                    if entities:
                        st.subheader("Medical Entities Found:")
                        st.write(", ".join(entities))

                except Exception as e:
                    st.error(f"An error occurred: {e}. Please try again.")

        else:
            st.warning("Please enter a question to get an answer.")

# Run the app
if __name__ == "__main__":
    main()
