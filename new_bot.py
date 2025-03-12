from flask import Flask, request, jsonify

app = Flask(__name__)

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
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    file1_path = 'page_corpus_v2.txt'
    context = read_file(file1_path)
    answer = get_answer_from_openai(question, context)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)