# app.py

import streamlit as st
import torch
import requests
import re
from youtube_transcript_api import YouTubeTranscriptApi
from rpunct import RestorePuncts 
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
        return None

def get_punctuation_model(language='en'):
    """Loads and returns the appropriate punctuation model."""
    if language == 'en':
        try:
            st.write(f"Attempting to load RPunct on CPU for language: {language}") 
            rp = RestorePuncts(device='cpu')
            st.write("RPunct model loaded successfully.") 
            return rp, 'rpunct'
        except Exception as e:
            st.write(f"RPunct loading error: {e}")
            st.warning(f"Could not load rpunct model (English): {e}. Proceeding without punctuation.")
            return None, None

    else:
        st.warning(f"Punctuation is currently only supported for English. Proceeding without punctuation for language: '{language}'.")
        return None, None

def punctuate_text(text, model, model_type):
    """Applies punctuation using the loaded model."""
    if model is None or model_type is None:
        return text

    if model_type == 'rpunct':
        try:
            result = model.punctuate(text)
            return result
        except Exception as e:
            st.error(f"Error during English punctuation with RPunct: {e}")
            return text
    else:
        return text

# --- Main Streamlit App Logic ---

st.set_page_config(page_title="YouTube Video Summarizer", layout="wide")
st.title("📺 YouTube Video Summarizer")
st.markdown("Enter a YouTube video URL to get its transcript summarized.")

try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except FileNotFoundError:
    api_key = st.text_input("Enter your OpenRouter API Key:", type="password", key="api_key_input")
except Exception as e: 
    st.error(f"Error accessing secrets: {e}")
    api_key = st.text_input("Enter your OpenRouter API Key (Secrets Error):", type="password", key="api_key_input_fallback")


# Initialize OpenAI Client
client = None
if api_key:
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    except Exception as e:
        st.error(f"Failed to initialize API Client: {e}")
        client = None
else:
    st.warning("Please provide your OpenRouter API Key.")


youtube_url = st.text_input("Enter YouTube Video URL:", key="youtube_url_input")

# Language Selection
lang_options = {"English": "en", "Indonesian": "id"}
selected_lang_name = st.selectbox("Select Language that used in the Video for Transcript:", options=list(lang_options.keys()), key="lang_select")
selected_lang_code = lang_options[selected_lang_name]

# Summarization Model
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
                    transcript_list = YouTubeTranscriptApi.list_transcripts(vid_id)
                    target_langs = [selected_lang_code, 'en'] if selected_lang_code != 'en' else ['en']
                    # Try fetching the selected language first, then English as fallback
                    try:
                        transcript = transcript_list.find_transcript(target_langs)
                    except Exception: 
                        transcript = transcript_list.find_generated_transcript(target_langs)

                    transcript_data = transcript.fetch()
                    actual_lang = transcript.language_code

                st.success(f"Transcript fetched successfully in '{actual_lang}'.")

                # Join transcript text
                transcript_joined = " ".join([line.text for line in transcript_data]) 

                # Display Raw Transcript 
                with st.expander("Show Raw Transcript"):
                    st.text_area("Raw Transcript", transcript_joined, height=200)

                # Punctuating Text (Only if English)
                punctuated_text = transcript_joined 
                punc_model, punc_model_type = get_punctuation_model(actual_lang)

                if punc_model and punc_model_type: # Check if a model was successfully loaded
                    with st.spinner(f"Punctuating text ({actual_lang})... This might take a moment."):
                        punctuated_text = punctuate_text(transcript_joined, punc_model, punc_model_type)

                # Display Punctuated Text
                with st.expander("Show (Potentially) Punctuated Transcript"):
                    st.text_area("Punctuated Transcript", punctuated_text, height=300)

                # Summarizing using LLM
                with st.spinner(f"Summarizing using {llm_model}..."):
                    try:
                        # Use the punctuated text for summarization
                        text_to_summarize = punctuated_text

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
                                    "content": f"Please summarize this transcript obtained from a YouTube video:\n\n---\n{text_to_summarize}\n---\n\n"
                                               "Focus on the core message, arguments, and conclusions presented."
                                }
                            ]
                        )
                        summary = completion.choices[0].message.content
                        st.subheader("Video Summary:")
                        st.markdown(summary)

                    except Exception as e:
                        st.error(f"LLM Summarization Error: {e}")
                        st.error("Could not generate summary. Please check the transcript length, API key, or model availability.")

            except YouTubeTranscriptApi.TranscriptsDisabled:
                 st.error("Transcripts are disabled for this video.")
            except YouTubeTranscriptApi.NoTranscriptFound:
                 st.error(f"No transcript found for video ID '{vid_id}' in the requested languages ({target_langs}).")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.error("Could not process the video. Please double-check the URL and language availability.")