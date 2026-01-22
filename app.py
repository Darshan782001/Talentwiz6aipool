from flask import Flask, request, jsonify, render_template, session, Response, redirect
from flask_cors import CORS
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
CORS(app)  # Enable CORS for all routes

# Configure Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME', 'Phi-4-mini-instruct')

# Configure Azure Storage
AZURE_STORAGE_ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT_NAME', 'qageneratorhistory')
AZURE_STORAGE_ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')

print(f"Loaded endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f"Loaded deployment: {AZURE_DEPLOYMENT_NAME}")

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')

# Initialize Azure Storage
try:
    from azure.storage.blob import BlobServiceClient
    blob_service_client = BlobServiceClient(
        account_url=f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=AZURE_STORAGE_ACCOUNT_KEY
    )
    print("Azure Storage initialized successfully")
except Exception as e:
    print(f"Azure Storage initialization failed: {e}")
    blob_service_client = None

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
        # Remove any markdown code blocks
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        # Clean up common issues
        text = text.strip()
        text = re.sub(r'^[^{]*', '', text)  # Remove text before first {
        text = re.sub(r'}[^}]*$', '}', text)  # Remove text after last }
        
        return json.loads(text)
def save_to_azure_storage(data):
    """Save QA session data to Azure Storage"""
    if not blob_service_client:
        return
    
    try:
        import uuid
        container_name = "qa-history"
        blob_name = f"qa-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}.json"
        
        # Create container if it doesn't exist
        try:
            blob_service_client.create_container(container_name)
        except:
            pass  # Container already exists
        
        # Upload data
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(json.dumps(data), overwrite=True)
        print(f"Saved QA session to Azure Storage: {blob_name}")
    except Exception as e:
        print(f"Error saving to Azure Storage: {e}")

def load_from_azure_storage():
    """Load QA history from Azure Storage"""
    if not blob_service_client:
        return []
    
    try:
        container_name = "qa-history"
        container_client = blob_service_client.get_container_client(container_name)
        
        history = []
        blobs = container_client.list_blobs()
        
        # Get all sessions, sorted by last modified
        blob_list = sorted(blobs, key=lambda x: x.last_modified, reverse=True)
        
        for blob in blob_list:
            try:
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
                data = json.loads(blob_client.download_blob().readall())
                # Add blob name for retrieval
                data['blob_name'] = blob.name
                history.append(data)
            except Exception as e:
                print(f"Error reading blob {blob.name}: {e}")
        
        return history
    except Exception as e:
        print(f"Error loading from Azure Storage: {e}")
        return []

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
        print(f"Azure API Response: {content[:200]}...")
        
        # Use improved JSON extraction
        return extract_json_from_text(content)
            
    except Exception as e:
        print(f"Azure OpenAI Error: {str(e)}")
        # Return fallback only if all retries fail
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

        def call_azure():
            return call_azure_openai(prompt)
        
        result = retry_with_backoff(call_azure)
        
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
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    try:
        # Get parameters from request
        job_description = request.json.get('jobDescription')
        experience_level = request.json.get('experienceLevel')
        skill_level = request.json.get('skillLevel')
        question_type = request.json.get('questionType')
        
        print(f"[{request_id}] Request received - Experience: {experience_level}, Skill: {skill_level}, Question Type: {question_type}")

        # Validation
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        if not experience_level:
            return jsonify({'error': 'Experience level is required'}), 400
        if not skill_level:
            return jsonify({'error': 'Skill level is required'}), 400
        if not question_type:
            return jsonify({'error': 'Question type is required'}), 400

        print(f"[{request_id}] Processing job description with experience: {experience_level}, skill: {skill_level}, question type: {question_type}")

        # Adjust question distribution based on question type
        if question_type == 'technical':
            question_distribution = """
Question Categories (distribute across 16 questions):
- Technical Skills (12 questions): Deep dive into specific technologies/tools, frameworks, and domain knowledge
- Problem Solving (2 questions): Technical challenges and solutions
- System Design (2 questions): Architecture and scalability considerations
"""
        elif question_type == 'technical-scenario':
            question_distribution = """
Question Categories (distribute across 16 questions):
- Technical Skills (8 questions): Core technologies, tools, and frameworks
- Scenario-based Technical (6 questions): Real-world technical scenarios and problem-solving
- System Design (2 questions): Architecture and scalability considerations
"""
        elif question_type == 'technical-coding':
            question_distribution = """
Question Categories (distribute across 16 questions):
- Technical Skills (6 questions): Core technologies, tools, and frameworks
- Coding Questions (8 questions): Write code snippets, solve algorithms, debug code, explain data structures, code optimization
- Best Practices (2 questions): Code quality, testing, performance optimization

For coding questions, include:
- Algorithm implementation questions
- Data structure problems
- Code debugging scenarios
- Code review and optimization
- Programming logic challenges
"""
        elif question_type == 'behavioral':
            question_distribution = """
Question Categories (distribute across 16 questions):
- Behavioral (8 questions): Leadership, teamwork, communication, conflict resolution
- Problem Solving (4 questions): Real-world scenarios and challenges
- Best Practices (2 questions): Code quality, security, performance
- Industry Knowledge (2 questions): Trends, future outlook
"""
        elif question_type == 'competency-based':
            question_distribution = """
Question Categories (distribute across 16 questions):
- Competency-based (10 questions): Specific skills, achievements, and experiences
- Problem Solving (3 questions): Real-world scenarios and challenges
- Best Practices (2 questions): Quality, efficiency, security considerations
- Leadership/Collaboration (1 question): Team dynamics, stakeholder management
"""
        elif question_type == 'situational':
            question_distribution = """
Question Categories (distribute across 16 questions):
- Situational (10 questions): "What would you do if..." scenarios
- Problem Solving (4 questions): Real-world scenarios and challenges
- System Design (2 questions): Architecture and scalability considerations
"""
        elif question_type == 'skill-based':
            question_distribution = """
Question Categories (distribute across 16 questions):
- Skill-based (12 questions): Specific technical skills, tools, and methodologies
- Problem Solving (2 questions): Technical challenges and solutions
- Best Practices (2 questions): Quality, efficiency, security considerations
"""
        else:  # mixed (all question types)
            question_distribution = """
Question Categories (distribute across 16 questions):
- Technical Skills (6 questions): Deep dive into specific technologies/tools
- Problem Solving (3 questions): Real-world scenarios and challenges
- System Design (2 questions): Architecture and scalability considerations
- Best Practices (2 questions): Code quality, security, performance
- Behavioral (2 questions): Leadership, teamwork, communication
- Industry Knowledge (1 question): Trends, future outlook
"""

        prompt = f"""You are a senior technical recruiter and interview architect. Generate exactly 16 comprehensive interview questions with detailed answers.

Job Description: {job_description}
Experience Level: {experience_level}
Skill Level: {skill_level}
Question Type Focus: {question_type}

Instructions:
1. Extract key technical skills, tools, frameworks, and domain knowledge from the JD
2. Create questions that test both technical depth and practical application
3. Include scenario-based questions that mirror real job challenges
4. Balance technical competency with behavioral and problem-solving skills
5. Ensure questions are appropriate for {experience_level} professionals with {skill_level} skill level
6. Focus on {question_type} style questions
7. Include questions about:
   - Core technical skills mentioned in JD
   - System design/architecture (for senior roles)
   - Problem-solving scenarios
   - Best practices and optimization
   - Team collaboration and leadership
   - Industry trends and continuous learning

{question_distribution}

Return ONLY valid JSON with detailed, professional answers:

{{
  "questions": [
    {{"question": "Technical question with specific context?", "answer": "Comprehensive answer covering key concepts, best practices, and real-world applications. Include specific examples and demonstrate deep understanding."}}
  ]
}}

Generate the JSON now:"""
        
        def call_azure():
            return call_azure_openai(prompt)
        
        result = retry_with_backoff(call_azure)
        

        
        print(f"[{request_id}] Generated {len(result.get('questions', []))} questions")
        
        # Save to Azure Storage
        storage_data = {
            'session_id': request_id,
            'job_description': job_description[:500],
            'experience_level': experience_level,
            'skill_level': skill_level,
            'question_type': question_type,
            'questions': result.get('questions', []),
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id
        }
        save_to_azure_storage(storage_data)
        
        # Store in Firestore if available
        if db:
            try:
                store_data = {
                    'job_description': job_description[:500],
                    'experience_level': experience_level,
                    'skill_level': skill_level,
                    'question_type': question_type,
                    'questions': result.get('questions', []),
                    'timestamp': datetime.now()
                }
                db.collection('qa_sessions').add(store_data)
            except Exception as e:
                print(f"Firestore error: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Generate QA Error: {str(e)}")
        return jsonify({'error': 'Failed to generate questions. Please try again.', 'details': str(e)}), 500


@app.route('/api/analyze-call', methods=['POST'])
def analyze_call():
    try:
        # Handle both JSON and form data
        if request.is_json:
            transcript = request.json.get('transcript')
            interview_id = request.json.get('interview_id')
            jd_text = request.json.get('jd', '')
        else:
            # Handle file upload
            jd_text = request.form.get('jd', '')
            interview_id = request.form.get('interview_id')
            audio_file = request.files.get('audio')
            
            if not audio_file:
                return jsonify({'error': 'No audio file provided'}), 400
            
            # Mock transcript extraction (replace with actual speech-to-text)
            transcript = f"Mock transcript from {audio_file.filename}. This would contain the actual conversation content from the audio file."
        
        if not transcript:
            return jsonify({'error': 'No transcript provided'}), 400
        
        prompt = f"""{SYSTEM_PROMPT}

Analyze this interview transcript against the job description:

Job Description: {jd_text}
Transcript: {transcript}

Return detailed JSON with:
{{
  "transcript": "{transcript[:500]}...",
  "nbro": "Overall recommendation for next round",
  "analysis": {{
    "sentiment_analysis": {{
      "overall_score": 75,
      "confidence_level": 85
    }},
    "engagement_metrics": {{
      "engagement_score": 80,
      "communication_clarity": 78
    }},
    "skills": [
      {{
        "skill": "Technical Knowledge",
        "score": 82,
        "feedback": "Strong understanding demonstrated",
        "recommendations": ["Continue technical deep-dive"]
      }}
    ],
    "summary": "Comprehensive analysis summary"
  }}
}}"""

        def call_azure():
            return call_azure_openai(prompt)
        
        result = retry_with_backoff(call_azure)
        
        # Generate interview ID if not provided
        if not interview_id:
            interview_id = f"call_{int(time.time())}"
        
        # Store analysis in Firestore
        if db:
            try:
                db.collection('call_analyses').document(interview_id).set({
                    'jd_text': jd_text[:500],
                    'transcript': transcript[:1000],
                    'analysis': result,
                    'interview_id': interview_id,
                    'timestamp': datetime.now()
                })
            except Exception as e:
                print(f"Firestore error: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-analysis')
def get_analysis():
    try:
        interview_id = request.args.get('interview_id')
        if not interview_id:
            return jsonify({'error': 'Interview ID required'}), 400
        
        if db:
            doc = db.collection('call_analyses').document(interview_id).get()
            if doc.exists:
                return jsonify(doc.to_dict())
        
        return jsonify({'error': 'Analysis not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analysis-history')
def analysis_history():
    return render_template('analysis_history.html')

@app.route('/api/analysis-history')
def get_analysis_history():
    try:
        if not db:
            return jsonify({'history': []})
        
        analyses = db.collection('call_analyses').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(20).get()
        
        history = []
        for analysis in analyses:
            data = analysis.to_dict()
            history.append({
                'interview_id': data.get('interview_id'),
                'timestamp': data.get('timestamp'),
                'jd_preview': data.get('jd_text', '')[:100] + '...' if data.get('jd_text') else 'No JD',
                'analysis_summary': data.get('analysis', {}).get('summary', 'No summary')[:100] + '...'
            })
        
        return jsonify({'history': history})
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

@app.route('/api/qa-history/<session_id>')
def get_qa_session(session_id):
    """Get specific QA session details"""
    try:
        if not blob_service_client:
            return jsonify({'error': 'Storage not available'}), 500
        
        container_name = "qa-history"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=session_id)
        
        data = json.loads(blob_client.download_blob().readall())
        return jsonify(data)
    except Exception as e:
        print(f"Error getting QA session: {e}")
        return jsonify({'error': 'Session not found'}), 404

@app.route('/api/qa-history')
def get_qa_history():
    """Get Q&A generation history from Azure Storage"""
    try:
        # Load from Azure Storage
        history = load_from_azure_storage()
        
        # Format for frontend
        formatted_history = []
        for item in history:
            formatted_history.append({
                'session_id': item.get('blob_name', item.get('session_id', item.get('request_id', ''))),
                'experience_level': item.get('experience_level', ''),
                'skill_level': item.get('skill_level', ''),
                'question_type': item.get('question_type', ''),
                'question_count': len(item.get('questions', [])),
                'timestamp': item.get('timestamp')
            })
        
        return jsonify({'history': formatted_history})
    except Exception as e:
        print(f"Error getting QA history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/customize-qa', methods=['POST', 'OPTIONS'])
def customize_qa():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    print("Customize QA endpoint called")  # Debug log
    try:
        # Simple test response first
        return jsonify({
            'questions': [
                {"question": "Test question 1?", "answer": "Test answer 1"},
                {"question": "Test question 2?", "answer": "Test answer 2"},
                {"question": "Test question 3?", "answer": "Test answer 3"}
            ]
        })
        
    except Exception as e:
        print(f"Error in customize_qa: {str(e)}")
        return jsonify({'error': 'Failed to customize questions', 'details': str(e)}), 500


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

        def call_azure():
            return call_azure_openai(prompt)
        
        result = retry_with_backoff(call_azure)
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)