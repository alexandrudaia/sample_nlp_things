#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import requests


# In[3]:


url='https://deliver.latoken.com/about'


# In[4]:


from bs4 import BeautifulSoup
import urllib
import urllib.request
import warnings
warnings.filterwarnings('ignore')


# In[5]:


html=urllib.request.urlopen(url).read()
html=html.decode('utf8')
soup=html
 
soup = BeautifulSoup(html, 'html.parser')


# In[10]:


canva1='https://www.canva.com/design/DAFmiiHpO7Q/view?embed'


# In[3]:


import openai
from youtube_transcript_api import YouTubeTranscriptApi

# Function to get video transcripts
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Combine text pieces into a single content block
        text = " ".join([t['text'] for t in transcript])
        return text
    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        return None

# Function to ask a question and get an answer using OpenAI's GPT-3
def ask_question(question, documents, conversation_history):
    # Prepare the context from the conversation history
    context = " ".join(["Q: " + qa['q'] + " A: " + qa['a'] for qa in conversation_history])
    prompt = context + " " + question
    
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,
        stop=None,
        temperature=0.5
    )
    return response.choices[0].text.strip()

# Main function to run the Q&A system
def main():
    video_ids = ['aVawpb2H3vg']  # Replace with your actual YouTube video IDs
    documents = []
    openai.api_key=' '
    
    for vid in video_ids:
        text = get_transcript(vid)
        if text:
            documents.append(text)
    
    # Initialize conversation history
    conversation_history = []

    # Example questions
    questions = [
        "What is the main topic discussed in the first video?",
        "Can you give more details on that topic?",
        "What is video authlor name ?"
    ]

    for question in questions:
        answer = ask_question(question, documents, conversation_history)
        print("Q:", question)
        print("A:", answer)
        # Append the current Q&A to the history
        conversation_history.append({'q': question, 'a': answer})

if __name__ == "__main__":
    main()


# In[3]:


pip install youtube_transcript_api


# In[13]:


import openai
import moviepy.editor as mp
import speech_recognition as sr

# Function to extract audio from video and convert it to text
def video_to_text(video_path):
    try:
        # Load video file and extract audio
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        audio_path = "temp_audio.wav"
        audio.write_audiofile(audio_path)
        
        # Use speech recognition to convert audio to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"Error processing video file: {e}")
        return None

# Function to read content from a text file
def read_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print("The file was not found.")
        return None
    except IOError:
        print("Error occurred while reading the file.")
        return None

# Function to ask a question and get an answer using OpenAI's GPT-3
def ask_question(question, documents, conversation_history):
    # Prepare the context from the conversation history
    context = " ".join(["Q: " + qa['q'] + " A: " + qa['a'] for qa in conversation_history])
    prompt = context + " " + question
    
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,
        stop=None,
        temperature=0.1
    )
    return response.choices[0].text.strip()

# Main function to run the Q&A system
def main():
    video_path = '":"https://video-private-assets.canva.com/VAFm1yLTPEs/v/b75f62ece6.mp4?exp=1716220620000&cf-ck=p05TDTP8uTUMRrhbkQ1rWbXEaoTR6XwnAgDBM7YmZo8&cf-sig=dew3zZD6sfih511HFQ1xixEaI-kmHc3hT-CCwBd4_5A&cf-sig-kid=CO7cCjZ_YiI=&sig=B3-NZIPojJ3cdz43iAIfYPq893aCt_x71elwVC3o8rI&sig-kid=GzFgFdhXD'
    documents = []
    openai.api_key='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'
    
    
    # Read additional text file
    text_file_content = read_text_file('hkttext.txt')  # Replace with your actual file path
    if text_file_content:
        documents.append(text_file_content)

    # Initialize conversation history
    conversation_history = []

    # Example questions
    questions = [
        "What hackaton does latoken has now ?",
       
    ]

    for question in questions:
        answer = ask_question(question, documents, conversation_hist)
        print("Q:", question)
        print("A:", answer)
        # Append the current Q&A to the history
        conversation_history.append({'q': question, 'a': answer})
    print(documents)

if __name__ == "__main__":
    main()


# In[20]:


documents


# In[25]:


import openai

 
openai.api_key=''
 

def generate_answer_from_text_file(file_path, question):
    try:
        # Read the content of the text file
        with open(file_path, 'r') as file:
            file_content = file.read()

        # Define the prompt for the model
        prompt = f"Based on the following content, please answer the question:\n\n{file_content}\n\nQuestion: {question}"

        # Call the OpenAI API to generate a response
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Extract and print the generated answer
        answer = response.choices[0].text.strip()
        print(f"Question: {question}")
        print(f"Generated Answer: {answer}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = 'your_text_file.txt'  # Replace with the path to your text file
question = "What is the main topic discussed in the document?"  # Replace with your question
generate_answer_from_text_file('hkttext.txt', 'How many trading assets there are ?')


# In[ ]:


pip install python-telegram-bot openai

import openai
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Set your API keys here
openai.api_key = 'your-openai-api-key-here'
TELEGRAM_BOT_TOKEN = 'your-telegram-bot-token-here'

# Path to your text file
TEXT_FILE_PATH = 'your_text_file.txt'

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\! Send me a question and I will answer it based on the content of the text file\.',
        reply_markup=ForceReply(selective=True),
    )

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Send me a question, and I will answer it based on the content of the text file.')

def generate_answer_from_text_file(file_path, question):
    try:
        # Read the content of the text file
        with open(file_path, 'r') as file:
            file_content = file.read()

        # Define the prompt for the model
        prompt = f"Based on the following content, please answer the question:\n\n{file_content}\n\nQuestion: {question}"

        # Call the OpenAI API to generate a response
        response = openai.Completion.create(
            engine="text-davinci-004",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Extract and return the generated answer
        answer = response.choices[0].text.strip()
        return answer

    except Exception as e:
        return f"An error occurred: {e}"

def handle_message(update: Update, context: CallbackContext) -> None:
    """Handle the user's message and generate an answer."""
    question = update.message.text
    answer = generate_answer_from_text_file(TEXT_FILE_PATH, question)
    update.message.reply_text(answer)

def main() -> None:
    """Start the bot."""
    updater = Updater(TELEGRAM_BOT_TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Register the command handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # Register the message handler
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT, SIGTERM, or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()

