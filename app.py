# app.py

import streamlit as st
import torch
import requests
import re  
from youtube_transcript_api import YouTubeTranscriptApi
from rpunct import RestorePuncts 
from transformers import pipeline
from openai import OpenAI


def get_vid_id(url_link):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)',  # Standard watch URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]+)', # Embed URL
        r'(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]+)',                 # Shortened URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]+)' # Shorts URL
    ]
    for pattern in patterns:
        match = re.search(pattern, url_link)
        if match:
            return match.group(1)
    try:
        return url_link.split("v=")[1].split("&")[0]
    except IndexError:
        return None # Indicate failure

def get_punctuation_model(language='en'):
    """Loads and returns the appropriate punctuation model."""
    if language == 'en':
        try:
            rp = RestorePuncts(device='cpu')
            return rp, 'rpunct'
        except Exception as e:
            st.warning(f"Could not load rpunct model (English): {e}. Punctuation might be basic.")
            return None, None
    elif language == 'id':
        try:
            punct_model = pipeline("text2text-generation", model="cahya/indonesian-t5-punctuation", device=-1)
            return punct_model, 't5'
        except Exception as e:
            st.warning(f"Could not load T5 model (Indonesian): {e}. Falling back to no punctuation.")
            return None, None
    else:
        # If language is neither English nor Indonesian
        st.warning(f"Punctuation not supported for language: {language}")
        return None, None

def punctuate_text(text, model, model_type):
    """Applies punctuation using the loaded model."""
    if model is None:
        return text # No punctuation if model failed to load or fallback chosen

    if model_type == 'rpunct':
        try:
            return model.punctuate(text)
        except Exception as e:
            st.error(f"Error during English punctuation: {e}")
            return text
    elif model_type == 't5':
        try:
            max_chunk_size = 512 # Check model's specific limit if needed
            if len(text.split()) > max_chunk_size * 0.8: # Heuristic check
                 st.warning("Transcript is long, T5 punctuation might be slow or incomplete.")
            # Run punctuation
            result = model(text, max_length=1024) # Adjust max_length if needed
            return result[0]['generated_text']
        except Exception as e:
            st.error(f"Error during Indonesian punctuation: {e}")
            return text 
    else:
        return text

# --- Main Streamlit App Logic ---

st.set_page_config(page_title="YouTube Video Summarizer", layout="wide")
st.title("📺 YouTube Video Summarizer")
st.markdown("Enter a YouTube video URL to get its transcript summarized. But keep in mind that this only works for Indonesian and English videos. ")


try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except FileNotFoundError:
    api_key = st.text_input("Enter your OpenRouter API Key:", type="password", key="api_key_input")

# Initialize OpenAI Client
# Check if API key is provided before initializing
client = None
if api_key:
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        st.success("API Client Initialized.")
    except Exception as e:
        st.error(f"Failed to initialize API Client: {e}")
        client = None # Ensure client is None if init fails
else:
    st.warning("Please provide your OpenRouter API Key.")


# YouTube URL Input
youtube_url = st.text_input("Enter YouTube Video URL:", key="youtube_url_input")

# Language Selection
lang_options = {"English": "en", "Indonesian": "id"}
selected_lang_name = st.selectbox("Select Language that used in the Video for Transcript:", options=list(lang_options.keys()), key="lang_select")
selected_lang_code = lang_options[selected_lang_name]

llm_model = "openrouter/quasar-alpha"


# Button to trigger processing
if st.button("Summarize Video", key="summarize_button", disabled=(not youtube_url or not client)):

    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
    elif not client:
        st.error("API Client not initialized. Please check your API Key.")
    else:
        vid_id = get_vid_id(youtube_url)

        if not vid_id:
            st.error("Could not extract Video ID from the URL. Please check the link.")
        else:
            st.info(f"Processing Video ID: {vid_id}")

            try:
                with st.spinner("Fetching transcript..."):
                    # Fetch transcript only in the selected language + English as fallback
                    transcript_list = YouTubeTranscriptApi.list_transcripts(vid_id)
                    target_langs = [selected_lang_code, 'en'] if selected_lang_code != 'en' else ['en']
                    transcript = transcript_list.find_generated_transcript(target_langs)
                    transcript_data = transcript.fetch()
                    actual_lang = transcript.language_code # Get the actual language fetched

                st.success(f"Transcript fetched successfully in '{actual_lang}'.")

                # Join transcript text
                transcript_joined = " ".join([line.text for line in transcript_data]) # Use space join

                # Display Raw Transcript (Optional)
                with st.expander("Show Raw Transcript"):
                    st.text_area("Raw Transcript", transcript_joined, height=200)

                # Punctuating Text
                punctuated_text = transcript_joined # Default if punctuation fails
                with st.spinner(f"Punctuating text ({actual_lang})... This might take a moment."):
                    # Load the appropriate model based on the *actual* fetched language
                    punc_model, punc_model_type = get_punctuation_model(actual_lang)
                    if punc_model:
                         punctuated_text = punctuate_text(transcript_joined, punc_model, punc_model_type)
                    else:
                         st.warning(f"Proceeding without punctuation for language '{actual_lang}'.")


                # Display Punctuated Text (Optional)
                with st.expander("Show Punctuated Transcript"):
                    st.text_area("Punctuated Transcript", punctuated_text, height=300)

                # Summarizing using LLM
                with st.spinner(f"Summarizing using {llm_model}..."):
                    try:
                        completion = client.chat.completions.create(
                            model=llm_model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant that summarizes YouTube video transcripts. "
                                               "Provide a concise yet comprehensive summary covering the main points and key takeaways. "
                                               "Structure the summary clearly, possibly using bullet points for key information."
                                },
                                {
                                    "role": "user",
                                    "content": f"Please summarize this transcript obtained from a YouTube video:\n\n---\n{punctuated_text}\n---\n\n"
                                               "Focus on the core message, arguments, and conclusions presented."
                                }
                            ]
                        )
                        summary = completion.choices[0].message.content
                        st.subheader("Video Summary:")
                        st.markdown(summary) # Use markdown for better formatting

                    except Exception as e:
                        st.error(f"LLM Summarization Error: {e}")
                        st.error("Could not generate summary. Please check the transcript length, API key, or model availability.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.error("Could not process the video. Possible reasons: Transcript not available for this video/language, invalid URL, network issue, or API error.")