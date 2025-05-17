# app.py

import streamlit as st
import torch
import requests
import re

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    RequestBlocked,  # Added
    IpBlocked,       # Added
    CouldNotRetrieveTranscript # Added
)
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

@st.cache_resource(show_spinner="Loading punctuation model...")
def get_punctuation_model(language='en'):
    if language == 'en':
        try:
            rp = RestorePuncts(device='cpu')
            return rp, 'rpunct'
        except Exception as e:
            st.error(f"Error loading punctuation model (RPunct): {e}")
            st.warning(f"Could not load rpunct model (English). Proceeding without punctuation.")
            return None, None
    else:
        st.warning(f"Punctuation is currently only supported for English. Proceeding without punctuation for language: '{language}'.")
        return None, None

def punctuate_text(text, model, model_type):
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
    st.info("OpenRouter API Key not found in Streamlit secrets. Please enter it manually.")
    api_key = st.text_input("Enter your OpenRouter API Key:", type="password", key="api_key_input")
except Exception as e:
    st.error(f"Error accessing Streamlit secrets: {e}")
    api_key = st.text_input("Enter your OpenRouter API Key (Secrets Error):", type="password", key="api_key_input_fallback")

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
    if "api_key_input" not in st.session_state or not st.session_state.api_key_input:
        st.warning("Please provide your OpenRouter API Key.")

youtube_url = st.text_input("Enter YouTube Video URL:", key="youtube_url_input")
lang_options = {"English": "en", "Indonesian": "id"}
selected_lang_name = st.selectbox("Select Language that used in the Video for Transcript:", options=list(lang_options.keys()), key="lang_select")
selected_lang_code = lang_options[selected_lang_name]
llm_model = "openrouter/quasar-alpha"

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

            ytt_api = YouTubeTranscriptApi()
            target_langs_priority = [] # Define for broader scope

            try:
                with st.spinner("Fetching transcript..."):
                    transcript_list = ytt_api.list(vid_id) # Use instance method
                    
                    target_langs_priority = [selected_lang_code]
                    if selected_lang_code != 'en':
                        target_langs_priority.append('en')

                    transcript_found_obj = None
                    try:
                        transcript_found_obj = transcript_list.find_transcript(target_langs_priority)
                    except NoTranscriptFound:
                        try:
                            transcript_found_obj = transcript_list.find_generated_transcript(target_langs_priority)
                        except NoTranscriptFound:
                            st.error(f"No transcript (manual or generated) found for video ID '{vid_id}' in the requested languages ({', '.join(target_langs_priority)}).")
                            # transcript_found_obj remains None

                    if transcript_found_obj:
                        transcript_data = transcript_found_obj.fetch()
                        actual_lang = transcript_found_obj.language_code
                        st.success(f"Transcript fetched successfully in '{actual_lang}'.")

                        transcript_joined = " ".join([line['text'] for line in transcript_data])

                        with st.expander("Show Raw Transcript"):
                            st.text_area("Raw Transcript", transcript_joined, height=200)

                        punctuated_text = transcript_joined
                        if actual_lang == 'en':
                            punc_model, punc_model_type = get_punctuation_model(actual_lang)
                            if punc_model and punc_model_type:
                                with st.spinner(f"Punctuating English text... This might take a moment."):
                                    punctuated_text = punctuate_text(transcript_joined, punc_model, punc_model_type)
                        else:
                            st.info(f"Punctuation is currently applied only to English transcripts. Transcript is in '{actual_lang}'.")

                        with st.expander("Show (Potentially) Punctuated Transcript"):
                            st.text_area("Punctuated Transcript", punctuated_text, height=300)

                        with st.spinner(f"Summarizing using {llm_model}..."):
                            try:
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
                                            "content": f"Please summarize this transcript obtained from a YouTube video (language: {actual_lang}):\n\n---\n{text_to_summarize}\n---\n\n"
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
                    # else: The case where transcript_found_obj is None is handled by the NoTranscriptFound error message above.

            except (RequestBlocked, IpBlocked) as e: # Specific catch for IP blocks
                st.error("🛑 Failed to retrieve transcript: YouTube is likely blocking or rate-limiting your IP address.")
                st.markdown(
                    "This can happen if the app is making too many requests, or if it's hosted on a common cloud provider IP (like Streamlit Cloud). "
                    "If you have control over the app's deployment environment (e.g., self-hosting), consider using a proxy. "
                    "For more information, refer to the `youtube-transcript-api` library's documentation on "
                    "[working around IP bans](https://github.com/jdepoix/youtube-transcript-api?tab=readme-ov-file#working-around-ip-bans-requestblocked-or-ipblocked-exception)."
                )
                st.error(f"Technical details from library: {e}")
            except TranscriptsDisabled:
                 st.error(f"Transcripts are disabled for the video: {vid_id}.")
            except NoTranscriptFound: # Fallback, though inner logic should catch most NoTranscriptFound cases
                 st.error(f"No transcript ultimately found for video ID '{vid_id}' with language preferences: {', '.join(target_langs_priority if target_langs_priority else ['English'])}.")
            except CouldNotRetrieveTranscript as e: # Catch other library-specific transcript errors
                st.error(f"A problem occurred while trying to retrieve the transcript: {e}")
                st.warning("Please ensure the video URL is correct, the video is public, and has transcripts available.")
            except Exception as e: # General fallback for other unexpected errors
                st.error(f"An unexpected error occurred during processing: {e}")
                st.error("Could not process the video. Please double-check the URL and language settings.")