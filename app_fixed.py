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
import requests
import re

load_dotenv()

app = Flask(__name__)

# Configure Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME', 'Phi-4-mini-instruct')

# Configure Gemini AI
try:
    import google.generativeai as genai
    genai.configure(api_key='AIzaSyBhdtnmWxm9JuCw7GTQbRCf0WBBaY5gNHc')
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    print(f"Gemini AI not available: {e}")
    model = None

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
        # First try direct JSON parsing
        return json.loads(text)
    except:
        try:
            # Look for JSON within code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Look for JSON without code blocks
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON found, create a fallback response
            return {"error": "Could not parse JSON response", "raw_response": text}
        except:
            return {"error": "Failed to extract JSON", "raw_response": text}

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
    """Call Azure OpenAI API with improved error handling"""
    try:
        # Fix the endpoint URL format
        if AZURE_OPENAI_ENDPOINT.endswith('/'):
            base_url = AZURE_OPENAI_ENDPOINT.rstrip('/')
        else:
            base_url = AZURE_OPENAI_ENDPOINT
            
        # Remove any existing path components and construct proper URL
        if '/openai/v1' in base_url:
            base_url = base_url.split('/openai/v1')[0]
        
        endpoint_url = f"{base_url}/openai/deployments/{AZURE_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_OPENAI_API_KEY
        }
        
        data = {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 2000,
            'temperature': 0.7
        }
        
        print(f"Calling Azure OpenAI at: {endpoint_url}")
        response = requests.post(endpoint_url, headers=headers, json=data, timeout=30)
        print(f"Azure API Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Azure API Error Response: {response.text}")
            raise Exception(f"Azure API returned status {response.status_code}: {response.text}")
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"Azure API Response Content: {content[:200]}...")
        
        # Use improved JSON extraction
        return extract_json_from_text(content)
        
    except Exception as e:
        print(f"Azure OpenAI Error: {str(e)}")
        # Return a fallback response instead of raising
        return {
            "error": f"Azure OpenAI API error: {str(e)}",
            "questions": [
                {
                    "question": "Tell me about your background and experience relevant to this role.",
                    "answer": "This is a general opening question to understand the candidate's background."
                },
                {
                    "question": "What interests you most about this position?",
                    "answer": "This helps gauge the candidate's motivation and interest level."
                },
                {
                    "question": "Describe a challenging project you've worked on recently.",
                    "answer": "This assesses problem-solving skills and technical experience."
                }
            ]
        }

def call_gemini_safe(prompt):
    """Call Gemini API with improved error handling"""
    try:
        if not model:
            raise Exception("Gemini model not available")
            
        response = model.generate_content(prompt)
        content = response.text
        print(f"Gemini Response: {content[:200]}...")
        
        # Use improved JSON extraction
        return extract_json_from_text(content)
        
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        # Return fallback response
        return {
            "error": f"Gemini API error: {str(e)}",
            "questions": [
                {
                    "question": "Tell me about your background and experience relevant to this role.",
                    "answer": "This is a general opening question to understand the candidate's background."
                },
                {
                    "question": "What interests you most about this position?",
                    "answer": "This helps gauge the candidate's motivation and interest level."
                },
                {
                    "question": "Describe a challenging project you've worked on recently.",
                    "answer": "This assesses problem-solving skills and technical experience."
                }
            ]
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-qa', methods=['POST'])
def generate_qa():
    try:
        method = request.json.get('method')
        
        if method == 'jd':
            job_description = request.json.get('jobDescription')
            if not job_description:
                return jsonify({'error': 'Job description is required'}), 400
                
            prompt = f"""Generate 8 targeted interview questions with detailed answers based on this job description.

Job Description: {job_description}

Create questions that specifically match the requirements and skills mentioned in the JD.
Include technical, behavioral, and scenario-based questions.

Return ONLY a valid JSON object in this exact format:
{{
  "questions": [
    {{
      "question": "question text here",
      "answer": "detailed sample answer here"
    }}
  ]
}}"""
            
            def call_azure():
                return call_azure_openai(prompt)
            
            result = retry_with_backoff(call_azure)
        
        elif method == 'title':
            job_title = request.json.get('jobTitle')
            experience_level = request.json.get('experienceLevel')
            job_description = request.json.get('jobDescription', '')
            
            if not job_title or not experience_level:
                return jsonify({'error': 'Job title and experience level are required'}), 400
            
            prompt = f"""Generate 8 interview questions with detailed answers for this role:

Role: {job_title}
Experience Level: {experience_level}
Additional Context: {job_description}

Create targeted questions appropriate for this role and experience level.
Include technical, behavioral, and scenario-based questions.

Return ONLY a valid JSON object in this exact format:
{{
  "questions": [
    {{
      "question": "question text here",
      "answer": "detailed sample answer here"
    }}
  ]
}}"""
            
            def call_gemini():
                return call_gemini_safe(prompt)
            
            result = retry_with_backoff(call_gemini)
        
        else:
            return jsonify({'error': 'Invalid method. Use "jd" or "title"'}), 400
        
        # Ensure we have questions in the result
        if 'questions' not in result or not result['questions']:
            result = {
                "questions": [
                    {
                        "question": "Tell me about your background and experience relevant to this role.",
                        "answer": "This is a general opening question to understand the candidate's background and how it relates to the position."
                    },
                    {
                        "question": "What interests you most about this position and our company?",
                        "answer": "This helps gauge the candidate's motivation, research, and genuine interest in the role."
                    },
                    {
                        "question": "Describe a challenging project you've worked on recently. How did you approach it?",
                        "answer": "This assesses problem-solving skills, technical experience, and methodology."
                    },
                    {
                        "question": "How do you handle working under pressure and tight deadlines?",
                        "answer": "This evaluates stress management and time management skills."
                    },
                    {
                        "question": "Tell me about a time when you had to learn a new technology or skill quickly.",
                        "answer": "This assesses adaptability and learning ability."
                    },
                    {
                        "question": "How do you approach collaboration with team members who have different working styles?",
                        "answer": "This evaluates teamwork and interpersonal skills."
                    },
                    {
                        "question": "Describe a situation where you had to make a difficult decision with limited information.",
                        "answer": "This assesses decision-making skills and ability to work with uncertainty."
                    },
                    {
                        "question": "Where do you see yourself professionally in the next 3-5 years?",
                        "answer": "This helps understand career goals and long-term fit with the organization."
                    }
                ]
            }
        
        # Store in Firestore if available
        if db:
            try:
                db.collection('qa_sessions').add({
                    'method': method,
                    'job_title': job_title if method == 'title' else None,
                    'experience_level': experience_level if method == 'title' else None,
                    'job_description': job_description[:500] if job_description else None,
                    'questions': result.get('questions', []),
                    'timestamp': datetime.now()
                })
            except Exception as e:
                print(f"Firestore error: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Generate QA Error: {str(e)}")
        return jsonify({
            'error': 'Failed to generate questions. Please try again.',
            'details': str(e)
        }), 500

# Keep all other routes from the original app
@app.route('/api/match', methods=['POST'])
def match_jd_resume():
    try:
        jd_text = request.form.get('jd_text')
        resume_file = request.files.get('resume')
        
        if not jd_text or not resume_file:
            return jsonify({'error': 'Missing JD text or resume file'}), 400
        
        # Extract text from different file formats
        filename = secure_filename(resume_file.filename)
        if filename.endswith('.pdf'):
            resume_text = extract_pdf_text(resume_file)
        elif filename.endswith('.docx'):
            resume_text = extract_docx_text(resume_file)
        else:
            resume_text = resume_file.read().decode('utf-8')
        
        prompt = f"""Compare this Job Description with the Resume and provide a matching analysis:

JD: {jd_text}
Resume: {resume_text}

Return ONLY a valid JSON object with: score (0-100), skill_gaps (array), strengths (array), cultural_fit (0-100)"""

        def call_gemini():
            return call_gemini_safe(prompt)
        
        result = retry_with_backoff(call_gemini)
        
        # Store in Firestore
        if db:
            try:
                db.collection('matches').add({
                    'jd_text': jd_text[:500],
                    'filename': filename,
                    'result': result,
                    'timestamp': datetime.now()
                })
            except Exception as e:
                print(f"Firestore error: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/matcher')
def matcher():
    return render_template('matcher.html')

@app.route('/qa-generator')
def qa_generator():
    return render_template('qa_generator.html')

@app.route('/voice-interview')
def voice_interview():
    return render_template('voice_interview.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/call-analysis')
def call_analysis():
    return render_template('call_analysis.html')

@app.route('/ai-mode')
def ai_mode():
    return render_template('ai_mode.html')

def extract_pdf_text(file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except:
        return "Error reading PDF file"

def extract_docx_text(file):
    try:
        doc = docx.Document(io.BytesIO(file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except:
        return "Error reading DOCX file"

if __name__ == '__main__':
    print("Starting TalentCore AI Application...")
    print(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
    print(f"Azure Deployment: {AZURE_DEPLOYMENT_NAME}")
    print(f"Gemini Available: {model is not None}")
    app.run(debug=True, host='0.0.0.0', port=5000)