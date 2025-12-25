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

load_dotenv()

app = Flask(__name__)

# Configure Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME', 'Phi-4-mini-instruct')

print(f"Loaded endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f"Loaded deployment: {AZURE_DEPLOYMENT_NAME}")

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

def retry_with_backoff(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt + random.uniform(0, 1))

def call_azure_openai(prompt):
    """Call Azure OpenAI API"""
    try:
        from openai import AzureOpenAI
        import html
        
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
        print(f"Azure API Response: {content}")
        
        # Decode HTML entities (fix &quot; to " etc.)
        content = html.unescape(content)
        
        # Additional cleanup for common HTML entities that might not be caught
        content = content.replace('&quot;', '"').replace('&#39;', "'").replace('&amp;', '&')
        
        # Clean up markdown code blocks if present
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        # Try to parse as JSON, if it fails, wrap in a basic structure
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Content that failed to parse: {content[:500]}...")
            # If response is not JSON, wrap it in a basic structure
            return {"response": content, "error": "Response was not valid JSON"}
            
    except Exception as e:
        print(f"Azure OpenAI Error: {str(e)}")
        raise e

@app.route('/')
def index():
    return render_template('index.html')

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
        
        prompt = f"""{SYSTEM_PROMPT}

Compare this Job Description with the Resume and provide a matching analysis:

JD: {jd_text}
Resume: {resume_text}

Return JSON with: score (0-100), skill_gaps (array), strengths (array), cultural_fit (0-100)"""

        def call_gemini():
            response = model.generate_content(prompt)
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {"response": response.text, "error": "Response was not valid JSON"}
        
        result = retry_with_backoff(call_gemini)
        
        # Store in Firestore
        if db:
            try:
                db.collection('matches').add({
                    'jd_text': jd_text[:500],  # Store first 500 chars
                    'filename': filename,
                    'result': result,
                    'timestamp': datetime.now()
                })
            except Exception as e:
                print(f"Firestore error: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-qa', methods=['POST'])
def generate_qa():
    try:
        method = request.json.get('method')
        
        if method == 'jd':
            job_description = request.json.get('jobDescription')
            prompt = f"""Analyze this job description and generate 8 targeted interview questions with answers:

Job Description: {job_description}

Generate questions that specifically match the requirements and skills mentioned in the JD.
Include technical, behavioral, and scenario-based questions.

Return JSON with:
{{
  "questions": [
    {{
      "question": "question text",
      "answer": "detailed sample answer"
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
            
            prompt = f"""Generate exactly 10 interview questions with detailed answers for this role:

Role: {job_title}
Experience Level: {experience_level}
Job Description: {job_description}

Create a mix of:
- 4 technical questions specific to the role
- 3 behavioral questions
- 2 scenario-based questions
- 1 culture fit question

IMPORTANT: You must return valid JSON in this exact format:
{{
  "questions": [
    {{
      "question": "What is your experience with [specific technology]?",
      "answer": "A detailed sample answer explaining the candidate's experience..."
    }},
    {{
      "question": "Tell me about a challenging project you worked on.",
      "answer": "A comprehensive answer describing a specific project..."
    }}
  ]
}}

Generate all 10 questions now:"""
            
            def call_azure():
                return call_azure_openai(prompt)
            
            result = retry_with_backoff(call_azure)
        
        else:
            return jsonify({'error': 'Invalid method'}), 400
        
        # Store in Firestore
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
        
        # Debug logging
        print(f"Generated {len(result.get('questions', []))} questions for method: {method}")
        if 'questions' in result:
            print(f"First question: {result['questions'][0] if result['questions'] else 'None'}")
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-call', methods=['POST'])
def analyze_call():
    try:
        transcript = request.json.get('transcript')
        interview_id = request.json.get('interview_id')
        
        prompt = f"""{SYSTEM_PROMPT}

Analyze this interview transcript for comprehensive insights:

{transcript}

Return detailed JSON with:
{{
  "sentiment_analysis": {{
    "overall_score": 0-100,
    "trend": "positive|neutral|negative",
    "confidence_level": 0-100,
    "emotional_markers": ["marker1", "marker2"]
  }},
  "engagement_metrics": {{
    "engagement_score": 0-100,
    "response_quality": 0-100,
    "communication_clarity": 0-100,
    "enthusiasm_level": 0-100
  }},
  "technical_assessment": {{
    "technical_competency": 0-100,
    "red_flags": ["flag1", "flag2"],
    "strengths": ["strength1", "strength2"],
    "knowledge_gaps": ["gap1", "gap2"]
  }},
  "behavioral_insights": {{
    "leadership_potential": 0-100,
    "problem_solving": 0-100,
    "cultural_fit": 0-100,
    "communication_style": "description"
  }},
  "recommendations": {{
    "hire_recommendation": "strong_yes|yes|maybe|no|strong_no",
    "next_steps": ["step1", "step2"],
    "areas_to_probe": ["area1", "area2"]
  }},
  "summary": "comprehensive summary"
}}"""

        def call_gemini():
            response = model.generate_content(prompt)
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {"response": response.text, "error": "Response was not valid JSON"}
        
        result = retry_with_backoff(call_gemini)
        
        # Store analysis in Firestore
        if db and interview_id:
            try:
                db.collection('interviews').document(interview_id).update({
                    'analysis': result,
                    'status': 'analyzed',
                    'analyzed_at': datetime.now()
                })
            except Exception as e:
                print(f"Firestore error: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/matcher')
def matcher():
    return render_template('matcher.html')

@app.route('/talentwiz')
def talentwiz():
    return redirect('https://talentwiz-bsh8f4a5gqhugae6.canadacentral-01.azurewebsites.net')

@app.route('/ats')
def ats():
    return redirect('https://talentwizats-a3endnhvb9a6cjeu.canadacentral-01.azurewebsites.net')

@app.route('/call-analysis')
def call_analysis():
    return render_template('call_analysis.html')

@app.route('/ai-mode')
def ai_mode():
    return render_template('ai_mode.html')

@app.route('/qa-generator')
def qa_generator():
    return render_template('qa_generator.html')

@app.route('/voice-interview')
def voice_interview():
    return render_template('voice_interview.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

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

# Voice Interview System
@app.route('/api/start-interview', methods=['POST'])
def start_interview():
    try:
        candidate_name = request.json.get('candidate_name')
        role = request.json.get('role')
        questions = request.json.get('questions', [])
        
        interview_id = f"interview_{int(time.time())}"
        session['interview_id'] = interview_id
        session['current_question'] = 0
        session['questions'] = questions
        session['transcript'] = []
        
        # Store interview session
        if db:
            db.collection('interviews').document(interview_id).set({
                'candidate_name': candidate_name,
                'role': role,
                'questions': questions,
                'status': 'started',
                'timestamp': datetime.now()
            })
        
        return jsonify({
            'interview_id': interview_id,
            'status': 'started',
            'first_question': questions[0] if questions else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/next-question', methods=['POST'])
def next_question():
    try:
        interview_id = session.get('interview_id')
        current_q = session.get('current_question', 0)
        questions = session.get('questions', [])
        answer = request.json.get('answer', '')
        
        # Store answer
        session['transcript'].append({
            'question': questions[current_q]['question'] if current_q < len(questions) else '',
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        current_q += 1
        session['current_question'] = current_q
        
        if current_q >= len(questions):
            return jsonify({'status': 'completed', 'message': 'Interview completed'})
        
        return jsonify({
            'status': 'continue',
            'question': questions[current_q],
            'progress': f"{current_q + 1}/{len(questions)}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice-stream', methods=['POST'])
def voice_stream():
    try:
        text = request.json.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Simulate TTS response (replace with actual Gemini TTS when available)
        response_data = {
            'text': text,
            'status': 'ready',
            'audio_url': f'/api/tts-audio?text={text[:50]}',  # Truncate for URL
            'duration': len(text) * 0.08
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard-stats')
def dashboard_stats():
    """Get analytics dashboard data"""
    try:
        if not db:
            return jsonify({'error': 'Database not available'}), 500
        
        # Get recent matches
        matches = db.collection('matches').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).get()
        
        dashboard_data = {
            'total_matches': len(matches),
            'avg_score': sum([m.to_dict().get('result', {}).get('score', 0) for m in matches]) / len(matches) if matches else 0,
            'recent_matches': [{
                'filename': m.to_dict().get('filename', 'Unknown'),
                'score': m.to_dict().get('result', {}).get('score', 0),
                'timestamp': m.to_dict().get('timestamp')
            } for m in matches]
        }
        
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/qa-history')
def get_qa_history():
    """Get Q&A generation history"""
    try:
        if not db:
            print("Firestore not available, returning empty history")
            return jsonify({'history': []})
        
        # Get recent Q&A sessions
        sessions = db.collection('qa_sessions').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(20).get()
        
        history = []
        for session in sessions:
            data = session.to_dict()
            print(f"Session data: {data}")  # Debug logging
            history.append({
                'job_title': data.get('job_title', 'Job Description'),
                'experience_level': data.get('experience_level', ''),
                'question_count': len(data.get('questions', [])),
                'timestamp': data.get('timestamp')
            })
        
        print(f"Returning {len(history)} history items")  # Debug logging
        return jsonify({'history': history})
    except Exception as e:
        print(f"Error getting QA history: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
# Hiring Manager AI Assistant
@app.route('/api/hiring-assistant', methods=['POST'])
def hiring_assistant():
    try:
        query = request.json.get('query')
        context = request.json.get('context', {})
        
        prompt = f"""{SYSTEM_PROMPT}

Act as a technical recruiting co-pilot. Answer this query: {query}

Context: {json.dumps(context)}

Provide structured assistance for:
- Candidate evaluation
- Interview feedback summaries  
- Hiring recommendations
- Process optimization

Return JSON with actionable insights."""

        def call_gemini():
            response = model.generate_content(prompt)
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {"response": response.text, "error": "Response was not valid JSON"}
        
        result = retry_with_backoff(call_gemini)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard-data')
def get_dashboard_data():
    """Enhanced dashboard with comprehensive analytics"""
    try:
        if not db:
            return jsonify({'error': 'Database not available'}), 500
        
        # Get analytics data
        matches = list(db.collection('matches').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).get())
        interviews = list(db.collection('interviews').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).get())
        qa_sessions = list(db.collection('qa_sessions').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).get())
        
        dashboard_data = {
            'overview': {
                'total_matches': len(matches),
                'total_interviews': len(interviews),
                'total_qa_sessions': len(qa_sessions),
                'avg_match_score': sum([m.to_dict().get('result', {}).get('score', 0) for m in matches]) / len(matches) if matches else 0
            },
            'recent_activity': [
                {
                    'type': 'match',
                    'filename': m.to_dict().get('filename', 'Unknown'),
                    'score': m.to_dict().get('result', {}).get('score', 0),
                    'timestamp': m.to_dict().get('timestamp')
                } for m in matches[:10]
            ],
            'interview_analytics': {
                'completed': len([i for i in interviews if i.to_dict().get('status') == 'analyzed']),
                'in_progress': len([i for i in interviews if i.to_dict().get('status') == 'started']),
                'avg_sentiment': sum([i.to_dict().get('analysis', {}).get('sentiment_analysis', {}).get('overall_score', 0) for i in interviews if i.to_dict().get('analysis')]) / len([i for i in interviews if i.to_dict().get('analysis')]) if interviews else 0
            },
            'role_distribution': {}
        }
        
        # Calculate role distribution
        for session in qa_sessions:
            role = session.to_dict().get('role', 'Unknown')
            dashboard_data['role_distribution'][role] = dashboard_data['role_distribution'].get(role, 0) + 1
        
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New routes for the enhanced UI
@app.route('/interview-generator')
def interview_generator():
    return render_template('interview_generator.html')

@app.route('/live-interview')
def live_interview():
    return render_template('live_interview.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/hiring-assistant')
def hiring_assistant_page():
    return render_template('hiring_assistant.html')