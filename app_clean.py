from flask import Flask, request, jsonify, render_template, session, Response, redirect
import json
import time
import random
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
import PyPDF2
import docx
import io
from datetime import datetime
import re

load_dotenv()

app = Flask(__name__)

# Configure Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME', 'Phi-4-mini-instruct')

print(f"Loaded endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f"Loaded deployment: {AZURE_DEPLOYMENT_NAME}")

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')

# Initialize Firestore (optional)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS_PATH', 'firebase-key.json'))
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Firestore initialization failed: {e}")
    db = None

SYSTEM_PROMPT = """You are TalentCore AI, a high-fidelity Talent Acquisition Intelligence agent. Your objective is to assist HR teams in 5 critical areas:

JD-Resume Matching: Provide neural-matching scores based on skills, seniority, and cultural markers.
Question Bank: Generate industry-standard Q&A for over 500+ specialized roles.
Voice Screening: Conduct natural-language interviews using TTS to evaluate technical and soft skills.
Call Analysis: Extract sentiment, technical red flags, and engagement scores from interview transcripts.
Act as a technical co-pilot for hiring managers.

Constraint: Always return data in structured JSON when requested. Maintain a neutral, professional, and data-driven tone."""

def extract_json_from_text(text):
    """Extract JSON from text that might contain markdown or other formatting"""
    try:
        return json.loads(text)
    except:
        try:
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            
            text = text.strip()
            text = re.sub(r'^[^{]*', '', text)
            text = re.sub(r'}[^}]*$', '}', text)
            
            return json.loads(text)
        except:
            return {
                "questions": [
                    {"question": "Tell me about your background.", "answer": "Sample answer about background."},
                    {"question": "What interests you about this role?", "answer": "Sample answer about interest."},
                    {"question": "Describe a challenging project.", "answer": "Sample answer about a project."}
                ]
            }

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt + random.uniform(0, 1))

def call_azure_openai(prompt):
    """Call Azure OpenAI API"""
    try:
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-02-15-preview"
        )
        
        completion = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        content = completion.choices[0].message.content
        print(f"Azure API Response: {content[:200]}...")
        
        return extract_json_from_text(content)
            
    except Exception as e:
        print(f"Azure OpenAI Error: {str(e)}")
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-qa', methods=['POST'])
def generate_qa():
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    try:
        method = request.json.get('method')
        print(f"[{request_id}] Request received - Method: {method}")
        
        if method == 'jd':
            job_description = request.json.get('jobDescription')
            experience_level = request.json.get('experienceLevel', '')
            if not job_description:
                return jsonify({'error': 'Job description is required'}), 400
                
            print(f"[{request_id}] JD Method - Processing job description with experience: {experience_level}")
            
            prompt = f"""You are an interview question generator. Generate exactly 8 interview questions with answers.

Job Description: {job_description}
Experience Level: {experience_level}

Rules:
1. Analyze the job description and experience level for appropriate questions
2. Create 8 questions suitable for {experience_level if experience_level else 'the role'}
3. Return ONLY valid JSON - no extra text
4. Use this exact format:

{{
  "questions": [
    {{"question": "Your question here?", "answer": "Sample answer here."}}
  ]
}}

Generate the JSON now:"""
            
            def call_azure():
                return call_azure_openai(prompt)
            
            result = retry_with_backoff(call_azure)
        
        elif method == 'title':
            job_title = request.json.get('jobTitle')
            experience_level = request.json.get('experienceLevel')
            job_description = request.json.get('jobDescription', '')
            
            print(f"[{request_id}] TITLE Method - Job: {job_title}, Experience: {experience_level}")
            
            if not job_title or not experience_level:
                return jsonify({'error': 'Job title and experience level are required'}), 400
            
            prompt = f"""You are an interview question generator. Generate exactly 8 interview questions with answers.

Role: {job_title}
Experience: {experience_level}
Context: {job_description}

Rules:
1. Make questions appropriate for {experience_level} experience
2. Return ONLY valid JSON - no extra text
3. Use this exact format:

{{
  "questions": [
    {{"question": "Your question here?", "answer": "Sample answer here."}}
  ]
}}

Generate the JSON now:"""
            
            def call_azure():
                return call_azure_openai(prompt)
            
            result = retry_with_backoff(call_azure)
        
        else:
            return jsonify({'error': 'Invalid method. Use "jd" or "title"'}), 400
        
        # Ensure we have questions in the result
        if 'questions' not in result or not result['questions'] or len(result['questions']) < 3:
            print(f"[{request_id}] API returned insufficient questions, using fallback")
            result = {
                "questions": [
                    {"question": "Tell me about your background and experience relevant to this role.", "answer": "This is a general opening question to understand the candidate's background."},
                    {"question": "What interests you most about this position?", "answer": "This helps gauge the candidate's motivation and interest level."},
                    {"question": "Describe a challenging project you've worked on recently.", "answer": "This assesses problem-solving skills and technical experience."}
                ]
            }
        else:
            print(f"[{request_id}] Successfully generated {len(result['questions'])} questions from API")
        
        print(f"[{request_id}] Generated {len(result.get('questions', []))} questions for method: {method}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Generate QA Error: {str(e)}")
        return jsonify({'error': 'Failed to generate questions. Please try again.', 'details': str(e)}), 500

@app.route('/customize-qa', methods=['POST'])
def customize_qa():
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    try:
        user_request = request.json.get('userRequest')
        current_questions = request.json.get('currentQuestions', [])
        context = request.json.get('context', {})
        
        print(f"[{request_id}] Customize request: {user_request}")
        
        prompt = f"""You are an interview question customization assistant. The user wants to modify interview questions.

User Request: {user_request}

Current Questions: {json.dumps(current_questions)}

Context: {json.dumps(context)}

Rules:
1. Understand what the user wants to change
2. Modify the questions accordingly
3. Keep the same number of questions unless specifically asked to change
4. Return ONLY valid JSON - no extra text
5. Use this exact format:

{{
  "questions": [
    {{"question": "Modified question here?", "answer": "Updated answer here."}}
  ]
}}

Generate the modified JSON now:"""
        
        def call_azure():
            return call_azure_openai(prompt)
        
        result = retry_with_backoff(call_azure)
        
        print(f"[{request_id}] Customized {len(result.get('questions', []))} questions")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Customize QA Error: {str(e)}")
        return jsonify({'error': 'Failed to customize questions. Please try again.', 'details': str(e)}), 500

@app.route('/qa-generator')
def qa_generator():
    return render_template('qa_generator.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)