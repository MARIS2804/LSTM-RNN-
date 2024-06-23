import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_my_model():
    return load_model('my_model.keras')

model = load_my_model()

# Assume token is predefined and loaded similarly
@st.cache_resource
def load_tokenizer():
    
    corpus="In the ever-accelerating dance of technological progress, Artificial Intelligence (AI) and Data Science have emerged as partners leading a revolution. AI, the aspiration of imbuing machines with human-like intelligence, has long captivated our imagination. Data science, the art of extracting knowledge from vast oceans of information, provides the fuel for AI advancements. Together, they are transforming industries, reshaping our understanding of the world, and pushing the boundaries of what possible.Data science acts as the AI eyes and ears, collecting and analyzing information from every corner of the digital world. From social media posts and financial transactions to scientific observations and medical records, data is the raw material from which AI learns and grows. Data scientists, the architects of this knowledge extraction, employ a diverse toolkit of statistical methods, machine learning algorithms, and programming languages to uncover hidden patterns, trends, and relationships within the data. This process involves cleaning and organizing the data, transforming it into a format suitable for analysis, and then building models that can identify patterns and make predictions.The insights gleaned from data science empower AI to make intelligent decisions and perform tasks that were once the exclusive domain of human expertise. Machine learning, a subfield of AI, allows computers to learn from data without explicit programming. By analyzing vast datasets, machine learning algorithms can identify complex patterns and relationships, enabling them to make predictions, classify information, and even generate creative content.One of the most captivating applications of AI lies in its ability to automate tasks. From self-driving cars navigating busy intersections to chatbots handling customer service inquiries, AI systems are streamlining processes and reducing human workload. In the realm of healthcare, AI algorithms are assisting doctors in analyzing medical images for early disease detection, while in finance, they are helping to identify fraudulent transactions and predict market trends.The power of AI extends beyond automation. AI systems are capable of deep learning, a technique inspired by the structure and function of the human brain. Deep learning algorithms consist of artificial neural networks, layered structures loosely mimicking the interconnected neurons in the brain. By training these networks on massive datasets, AI can achieve remarkable feats of recognition and understanding. Facial recognition software used for security purposes and natural language processing systems that power voice assistants are just a few examples of deep learning's capabilities.However, the burgeoning field of AI also presents challenges that demand careful consideration. One of the primary concerns is the issue of bias. If the data used to train AI systems is inherently biased, the resulting algorithms will perpetuate those biases, leading to discriminatory outcomes. Addressing bias in data and algorithms is crucial to ensure fairness and ethical use of AI.Another challenge lies in the interpretability of AI decisions.  While AI systems can make remarkably accurate predictions, understanding the reasoning behind those predictions can be difficult. This lack of transparency can raise concerns about accountability, particularly in high-stakes situations like autonomous vehicles or medical diagnosis. Efforts are underway to develop more transparent AI models, allowing humans to understand the rationale behind their decisions.The future of AI and data science is brimming with possibilities. As the volume and variety of data continue to grow exponentially, AI systems will become even more sophisticated, blurring the lines between human and machine intelligence. Advances in areas like quantum computing hold the promise of even more powerful AI capabilities.The potential applications of AI and data science are vast, encompassing nearly every facet of human life. From personalized education experiences and intelligent transportation systems to personalized medicine and environmental monitoring, these fields have the potential to create a more efficient, sustainable, and equitable future. However, navigating the ethical and social implications of AI will be paramount. Open dialogue, collaboration between experts, and robust regulatory frameworks are essential to ensure that AI serves the greater good and benefits all of humanity.The journey of AI and data science is far from over. As we continue to explore the frontiers of this groundbreaking technology, we must do so with a commitment to responsible development and ethical application. It is through this commitment that we can harness the immense potential of AI and data science to build a brighter future for ourselves and generations to come"
    token = Tokenizer()
    token.fit_on_texts([corpus])
    # Assuming you have a function to load your tokenizer
    # token = load_my_tokenizer_function()
    return token

token = load_tokenizer()

# Function to generate text predictions
def generate_text(input_text, no_of_predict, max_length):
    for _ in range(no_of_predict):
        in_seq = token.texts_to_sequences([input_text])[0]
        in_pad = pad_sequences([in_seq], maxlen=max_length-1)
        pred_seq = np.argmax(model.predict(in_pad), axis=1)
        pred_word = ""
        for word, index in token.word_index.items():
            if index == pred_seq:
                pred_word = word
                break
        input_text += " " + pred_word
    return input_text

# Streamlit app layout
st.title("Text Generation with TensorFlow")
st.write("This app generates text using a TensorFlow model.")

input_text = st.text_input("Enter initial text")
no_of_predict = st.number_input("Number of words to predict", min_value=1, step=1)
max_length = 32

if st.button("Generate Text"):
    if input_text and no_of_predict > 0 and max_length > 0:
        try:
            output_text = generate_text(input_text, no_of_predict, max_length)
            st.success(f"Generated Text: {output_text}")
        except Exception as e:
            st.write(f"Error: {e}")
    else:
        st.write("Please enter valid input text and parameters.")
