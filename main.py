import streamlit as st
from streamlit_mic_recorder import speech_to_text
from streamlit_TTS import auto_play, text_to_audio
from groq import Groq
import openai
import json

# Model names
MODEL_OPTIONS = {
    "Llama": "llama3-8b-8192",
    "Gemma": "gemma-7b-it",
    "Mistral": "mixtral-8x7b-32768",
    "OpenAI": "gpt-4o-mini"
}

# Language options
LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Marathi": "mr"
}

# Sidebar selection for model and language
st.sidebar.title("Settings")
selected_model_name = st.sidebar.radio("Choose the model:", list(MODEL_OPTIONS.keys()))
selected_language = st.sidebar.selectbox("Select Language:", list(LANGUAGE_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_model_name]
selected_language_code = LANGUAGE_OPTIONS[selected_language]

# Initialize session state for each model's chat history
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {model: [] for model in MODEL_OPTIONS.keys()}

# Retrieve the chat history for the selected model
chat_history = st.session_state.chat_histories[selected_model_name]

# Initialize Groq client and OpenAI API key
client = Groq(api_key="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")
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI key

# System prompt to guide the assistant's responses
system_prompt = {
    "role": "system",
    "content": (
        "You are a concerned and discerning parent being approached by a Byju's salesperson. "
        "Ask questions that a parent would realistically consider, such as course benefits, quality, "
        "teaching methods, and cost. Inquire about flexible payment options, any trial periods, and "
        "other parent testimonials. Be polite but thorough, ensuring this course is a good fit for your child."
        f"Make sure you generate your query strictly in the following language: {selected_language}. "
        "The output should follow this JSON Format strictly: {'query': query_generated}"
    )
}

# Function to call OpenAI API
def get_openai_response(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            top_p=1.0,
            stop=None,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message["content"]
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

# Function to get response from Groq API
def get_groq_response(messages, model):
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return None

def parse_query_from_response(response_text):
    try:
        # Parse JSON response to extract the 'query' field
        response_json = json.loads(response_text)
        return response_json.get("query", "")
    except json.JSONDecodeError:
        st.error("Error decoding JSON response.")
        return None

# Step 1: Record and transcribe audio in the selected language
st.write("### Record and Transcribe Salesperson's Speech:")
recorded_text = speech_to_text(
    language=selected_language_code,
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    use_container_width=True,
    just_once=True,
    key=f'STT_{selected_model_name}_{selected_language}'  # Unique key per model and language
)

# Step 2-4: Generate question, convert to speech, and play
if recorded_text:
    st.write("**Salesperson:**", recorded_text)
    chat_history.append({"role": "salesperson", "content": recorded_text})

    # Prepare message history
    messages = [system_prompt] + [
        {"role": "user" if message["role"] == "salesperson" else "assistant", "content": message["content"]}
        for message in chat_history
    ]

    # Get response based on selected model
    if selected_model_name == "OpenAI":
        response_text = get_openai_response(messages)
    else:
        response_text = get_groq_response(messages, selected_model)

    # Append the response to chat history if valid and play response as audio
    if response_text:
        # Parse the 'query' content from the response
        parsed_query = parse_query_from_response(response_text)
        
        if parsed_query:
            st.write("**Parent (AI):**", parsed_query)
            chat_history.append({"role": "assistant", "content": parsed_query})

            # Convert the parsed query to speech in the selected language and play
            response_audio = text_to_audio(parsed_query, language=selected_language_code)
            auto_play(response_audio)

# Show Chat History Button
if st.sidebar.button("Show Chat History"):
    st.sidebar.write("### Chat History")
    for entry in chat_history:
        role = "User" if entry["role"] == "salesperson" else "Assistant"
        st.sidebar.write(f"**{role}:** {entry['content']}")

# Save chat history regularly
st.session_state.chat_histories[selected_model_name] = chat_history
