# TalentCore AI - Talent Acquisition Intelligence Platform

A high-fidelity AI-powered recruitment system that assists HR teams with intelligent candidate matching, interview question generation, voice screening, and call analysis.

## Features

### 1. Neural Matcher
- AI-powered JD-Resume matching with neural scoring
- Skills gap analysis and strength identification
- Cultural fit assessment
- Support for PDF, DOCX, and TXT resume formats

### 2. Q&A Generator
- Generate industry-standard interview questions for 500+ specialized roles
- Generative retrieval system (no static files needed)
- Questions categorized by type and difficulty

### 3. Voice Interview
- Real-time voice screening with TTS integration
- Live transcript generation
- Natural language interview flow

### 4. Call Analysis
- Sentiment scoring and engagement metrics
- Technical red flag detection
- Comprehensive interview summaries

### 5. Analytics Dashboard
- Real-time recruitment metrics
- Historical match data
- Success rate tracking

## Tech Stack

- **Backend**: Python 3.9+ with Flask
- **AI Engine**: Google Gemini 2.0 Flash
- **Database**: Firebase Firestore
- **Frontend**: Jinja2 templates with Bootstrap 5
- **Deployment**: AWS Elastic Beanstalk / Heroku / App Runner

## Setup Instructions

### Prerequisites

1. Python 3.9 or higher
2. Google Gemini API key
3. Firebase project with Firestore enabled

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd talentcore-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:

Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your-gemini-api-key
FLASK_SECRET_KEY=your-secret-key
FLASK_ENV=development
FIREBASE_CREDENTIALS_PATH=firebase-key.json
```

4. Set up Firebase:
   - Create a Firebase project at https://console.firebase.google.com
   - Enable Firestore Database
   - Download service account credentials as `firebase-key.json`
   - Place the file in the project root

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Deployment

### AWS Elastic Beanstalk

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize EB:
```bash
eb init -p python-3.9 talentcore-ai
```

3. Create environment:
```bash
eb create talentcore-prod
```

4. Set environment variables:
```bash
eb setenv GEMINI_API_KEY=your-key FLASK_SECRET_KEY=your-secret
```

5. Deploy:
```bash
eb deploy
```

### Heroku

1. Create Heroku app:
```bash
heroku create talentcore-ai
```

2. Set environment variables:
```bash
heroku config:set GEMINI_API_KEY=your-key
heroku config:set FLASK_SECRET_KEY=your-secret
```

3. Deploy:
```bash
git push heroku main
```

## API Endpoints

### POST /api/match
Match JD with resume
- **Body**: `multipart/form-data` with `jd_text` and `resume` file
- **Response**: JSON with score, skill_gaps, strengths, cultural_fit

### POST /api/generate-qa
Generate interview questions
- **Body**: `{"role": "Cloud Architect"}`
- **Response**: JSON with questions array

### POST /api/voice-stream
Generate TTS audio
- **Body**: `{"text": "Interview question"}`
- **Response**: JSON with audio data

### POST /api/analyze-call
Analyze interview transcript
- **Body**: `{"transcript": "Interview text"}`
- **Response**: JSON with sentiment_score, technical_flags, engagement_score, summary

### GET /api/dashboard
Get analytics data
- **Response**: JSON with total_matches, avg_score, recent_matches

## Architecture

```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Flask    │ ◄─── Jinja2 Templates
│   Backend   │
└──────┬──────┘
       │
       ├──────► Gemini API (AI Processing)
       │
       └──────► Firestore (Data Storage)
```

## Security Features

- Exponential backoff with 5 retry attempts for 99.9% uptime
- Secure file upload handling with filename sanitization
- Environment-based configuration
- Session management for user state
- PII data truncation in database storage

## Performance

- Server-side rendering for fast initial load
- Minimal client-side JavaScript
- CDN-ready static assets
- Efficient database queries with pagination

## License

MIT License

## Support

For issues and questions, please open a GitHub issue.