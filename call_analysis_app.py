from flask import Flask, render_template, request, jsonify
import json
import time
import os
from dotenv import load_dotenv
from datetime import datetime
import threading
import tempfile
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
from openai import AzureOpenAI
import html

app = Flask(__name__)
load_dotenv()

# History management
HISTORY_FILE = 'history.json'
HISTORY_LOCK = threading.Lock()

def load_history():
    try:
        with HISTORY_LOCK:
            if not os.path.exists(HISTORY_FILE):
                return []
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def save_history(history):
    try:
        with HISTORY_LOCK:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

def add_interview_to_history(interview):
    history = load_history()
    history.insert(0, interview)
    save_history(history)

SYSTEM_PROMPT = """
You are TalentWiz, an expert AI assistant for technical interviews and candidate analysis. Provide structured, unbiased, and actionable insights based on the job description and interview transcript. Always respond in valid JSON as instructed.
"""

@app.route('/call-analysis')
def call_analysis():
    return render_template('call_analysis.html')

@app.route('/api/analyze-call', methods=['POST'])
def analyze_call():
    try:
        jd = request.form.get('jd')
        interview_id = request.form.get('interview_id')
        audio_file = request.files.get('audio')
        
        if not jd or not audio_file:
            return jsonify({'error': 'Missing JD or audio file'}), 400

        # Transcribe audio using Azure Speech-to-Text
        transcript = None
        try:
            speech_key = os.getenv('AZURE_SPEECH_KEY')
            speech_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
            
            # Save uploaded file to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as temp_in:
                audio_file.stream.seek(0)
                temp_in.write(audio_file.stream.read())
                temp_in.flush()
                temp_in_filename = temp_in.name

            # Convert to WAV PCM 16kHz mono
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                audio = AudioSegment.from_file(temp_in_filename)
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                audio.export(temp_wav.name, format="wav")
                temp_wav_filename = temp_wav.name

            # Azure Speech-to-Text
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
            speech_config.speech_recognition_language = "en-IN"
            audio_input = speechsdk.AudioConfig(filename=temp_wav_filename)
            
            conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
                audio_config=audio_input, speech_config=speech_config
            )
            
            all_results = []
            done = False
            
            def handle_final_result(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    all_results.append(evt.result.text)
            
            def stop_cb(evt):
                nonlocal done
                done = True
            
            conversation_transcriber.transcribed.connect(handle_final_result)
            conversation_transcriber.session_stopped.connect(stop_cb)
            conversation_transcriber.canceled.connect(stop_cb)
            conversation_transcriber.start_transcribing_async()

            # Wait for transcription with timeout
            timeout = 600
            start_time = time.time()
            while not done and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            conversation_transcriber.stop_transcribing_async()
            transcript = ' '.join(all_results)
            
            if not transcript:
                raise Exception("Speech-to-Text failed: No transcript recognized.")
            
            # Cleanup temp files
            try:
                os.remove(temp_in_filename)
                os.remove(temp_wav_filename)
            except Exception as cleanup_e:
                print(f"Warning: could not delete temp files: {cleanup_e}")
                
        except Exception as e:
            print(f"Azure Speech-to-Text error: {e}")
            return jsonify({'error': f'Audio transcription failed: {str(e)}'}), 500

        # Analyze transcript with Azure OpenAI
        prompt = f"""
{SYSTEM_PROMPT}

Job Description:
{jd}

Interview Transcript:
{transcript}

Analyze the above conversation and rate the candidate according to the primary skills required by the job description. For each primary skill, provide a score (0-100), specific feedback, and actionable recommendations. Also include overall sentiment, engagement, and a summary.

Most importantly, provide a clear and decisive field called "final_recommendation" with one of the following values:
- "Recommended for next round"
- "Maybe"
- "Not recommended"

Your response must be valid JSON in the following format:
{{
    "skills": [
        {{
            "skill": "string",
            "score": 0-100,
            "feedback": "string",
            "recommendations": ["string", ...]
        }}
    ],
    "sentiment_analysis": {{
        "overall_score": 0-100,
        "trend": "positive|neutral|negative",
        "confidence_level": 0-100,
        "emotional_markers": ["marker1", "marker2"]
    }},
    "engagement_metrics": {{
        "engagement_score": 0-100,
        "communication_clarity": 0-100,
        "enthusiasm_level": 0-100
    }},
    "summary": "comprehensive summary",
    "final_recommendation": "Recommended for next round|Maybe|Not recommended"
}}
Respond with only valid JSON. Do not include any extra text, markdown, or explanation.
"""

        client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT_CHATGPT5'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY_CHATGPT5'),
            api_version="2024-02-15-preview"
        )
        
        completion = client.chat.completions.create(
            model=os.getenv('AZURE_DEPLOYMENT_NAME_CHATGPT5'),
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        
        content = completion.choices[0].message.content
        content = html.unescape(content)
        content = content.replace('&quot;', '"').replace('&#39;', "'").replace('&amp;', '&')
        
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            return jsonify({
                'error': 'The AI response was not valid JSON. Please try again.',
                'raw_response': content
            }), 500

        # Calculate recommendations
        skills = result.get('skills', [])
        avg_score = sum([s.get('score', 0) for s in skills]) / len(skills) if skills else 0
        
        if avg_score >= 80:
            recommendation = 'Recommended for next round'
        elif avg_score >= 70:
            recommendation = 'Maybe'
        else:
            recommendation = 'Not recommended'

        overall_score = result.get('sentiment_analysis', {}).get('overall_score', 0)
        if overall_score >= 80:
            overall_recommendation = 'Recommended for next round'
        elif overall_score >= 70:
            overall_recommendation = 'Maybe'
        else:
            overall_recommendation = 'Not recommended'

        # Store in history
        if interview_id:
            interview_obj = {
                'interview_id': interview_id,
                'candidate_name': request.form.get('candidate_name', ''),
                'role': request.form.get('role', ''),
                'transcript': transcript,
                'analysis': result,
                'status': 'analyzed',
                'analyzed_at': datetime.now().isoformat(),
                'recommendation': recommendation,
                'manual_recommendation': '',
                'timestamp': datetime.now().isoformat()
            }
            add_interview_to_history(interview_obj)

        response = {
            'transcript': transcript,
            'analysis': result,
            'recommendation': recommendation,
            'manual_recommendation': '',
            'overall_recommendation': overall_recommendation,
            'overall_score': overall_score,
            'nbro': overall_recommendation
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print("Exception in analyze_call:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)