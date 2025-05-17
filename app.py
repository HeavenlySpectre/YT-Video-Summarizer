# app.py

import streamlit as st
import torch # Keep torch import if rpunct or its dependencies need it explicitly at global scope
import requests
import re
# Corrected import for youtube_transcript_api exceptions
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
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
    # Fallback, though the regex should cover most cases
    try:
        return url_link.split("v=")[1].split("&")[0]
    except IndexError:
        return None

@st.cache_resource(show_spinner="Loading punctuation model...") # Cache the model loading
def get_punctuation_model(language='en'):
    """Loads and returns the appropriate punctuation model."""
    if language == 'en':
        try:
            # st.write(f"Attempting to load RPunct on CPU for language: {language}") # Debug, can be removed
            rp = RestorePuncts(device='cpu')
            # st.write("RPunct model loaded successfully.") # Debug, can be removed
            return rp, 'rpunct'
        except Exception as e:
            # st.write(f"RPunct loading error: {e}") # Use st.error for actual errors for visibility
            st.error(f"Error loading punctuation model (RPunct): {e}")
            st.warning(f"Could not load rpunct model (English). Proceeding without punctuation.")
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
    st.info("OpenRouter API Key not found in Streamlit secrets. Please enter it manually.")
    api_key = st.text_input("Enter your OpenRouter API Key:", type="password", key="api_key_input")
except Exception as e:
    st.error(f"Error accessing Streamlit secrets: {e}")
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
    if "api_key_input" not in st.session_state or not st.session_state.api_key_input: # Avoid warning if field is empty but user hasn't submitted
        st.warning("Please provide your OpenRouter API Key.")


youtube_url = st.text_input("Enter YouTube Video URL:", key="youtube_url_input")

# Language Selection
lang_options = {"English": "en", "Indonesian": "id"}
selected_lang_name = st.selectbox("Select Language that used in the Video for Transcript:", options=list(lang_options.keys()), key="lang_select")
selected_lang_code = lang_options[selected_lang_name]

# Summarization Model
llm_model = "openrouter/quasar-alpha" # Consider making this configurable

# Button to trigger processing
if st.button("Summarize Video", key="summarize_button", disabled=(not youtube_url or not client)):

    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
    elif not client: # This check is slightly redundant due to 'disabled' but good for explicit feedback
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
                    # Define target languages for transcript fetching
                    # Try selected language, then English as fallback, then any generated transcript in selected or English
                    target_langs_priority = [selected_lang_code]
                    if selected_lang_code != 'en':
                        target_langs_priority.append('en')

                    transcript_found = False
                    try:
                        # Try finding a manually created transcript in the priority languages
                        transcript = transcript_list.find_transcript(target_langs_priority)
                        transcript_found = True
                    except NoTranscriptFound:
                        # If not found, try finding a generated transcript in the priority languages
                        try:
                            transcript = transcript_list.find_generated_transcript(target_langs_priority)
                            transcript_found = True
                        except NoTranscriptFound:
                            st.error(f"No transcript (manual or generated) found for video ID '{vid_id}' in the requested languages ({', '.join(target_langs_priority)}).")
                            # Optionally, you could try to fetch *any* available transcript as a last resort
                            # available_transcripts = transcript_list.find_manually_created_transcript(['en', 'id', ...all possible]) or find_generated_transcript
                            transcript = None # Ensure transcript is None if not found


                    if transcript:
                        transcript_data = transcript.fetch()
                        actual_lang = transcript.language_code
                        st.success(f"Transcript fetched successfully in '{actual_lang}'.")

                        # Join transcript text
                        transcript_joined = " ".join([line['text'] for line in transcript_data]) # Corrected access to line['text']

                        # Display Raw Transcript
                        with st.expander("Show Raw Transcript"):
                            st.text_area("Raw Transcript", transcript_joined, height=200)

                        # Punctuating Text (Only if English)
                        punctuated_text = transcript_joined
                        if actual_lang == 'en': # Only attempt punctuation if the actual transcript is English
                            punc_model, punc_model_type = get_punctuation_model(actual_lang) # 'en'
                            if punc_model and punc_model_type: # Check if a model was successfully loaded
                                with st.spinner(f"Punctuating English text... This might take a moment."):
                                    punctuated_text = punctuate_text(transcript_joined, punc_model, punc_model_type)
                        else:
                            st.info(f"Punctuation is currently applied only to English transcripts. Transcript is in '{actual_lang}'.")


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

            except TranscriptsDisabled:
                 st.error(f"Transcripts are disabled for the video: {vid_id}.")
            except NoTranscriptFound: 
                 st.error(f"No transcript found for video ID '{vid_id}' in the requested languages after all attempts.")
            except Exception as e:
                st.error(f"An unexpected error occurred during processing: {e}")
                st.error("Could not process the video. Please double-check the URL and language availability.")