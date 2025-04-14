import openai
from openai import OpenAI

import nest_asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set up your OpenAI API key
openai.api_key = ' 
client = OpenAI(
    api_key = ' 

def read_file(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()

def get_answer_from_openai(question, context):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
        ],
        max_tokens=512,
        temperature=0.13,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    answer =  response.choices[0].message.content
    return answer

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hi! Send me a question and I will answer it based on the provided context.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = update.message.text
    
    # Paths to your text files
    file1_path = 'page_corpus.txt'
    
    # Read the content of the text files
    file1_content = read_file(file1_path)
    
    # Combine the contents into a single context
    context_text = file1_content
    
    # Get the answer from OpenAI
    answer = get_answer_from_openai(question, context_text)
    
    # Send the answer back to the user
    await update.message.reply_text(answer)

def main() -> None:
    application = ApplicationBuilder().token("7846009421:AAHIxcPIEjbwF0t7f9x8M9469lHwZAJsCUI").build()

    # Add command handler for the /start command
    application.add_handler(CommandHandler("start", start))
    
    # Add message handler for text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
