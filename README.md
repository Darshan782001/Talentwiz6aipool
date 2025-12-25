# TalentCore AI - Talent Intelligence Suite

A dynamic AI-powered recruitment platform using Python/Flask and Gemini 2.5 Flash.

## Features

- **Neural Matcher**: AI-powered JD-Resume matching with skill gap analysis
- **Q&A Generator**: Generate interview questions for 500+ specialized roles  
- **Voice Interview**: Conduct AI-powered voice screening interviews
- **Call Analysis**: Extract sentiment and engagement scores from transcripts

## Quick Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables in `.env`:
```
GEMINI_API_KEY=your-gemini-api-key
FLASK_SECRET_KEY=your-secret-key
```

3. Run locally:
```bash
python app.py
```

## Deployment

### Heroku
```bash
git init
git add .
git commit -m "Initial commit"
heroku create your-app-name
heroku config:set GEMINI_API_KEY=your-key
git push heroku main
```

### AWS Elastic Beanstalk
```bash
eb init
eb create talentcore-env
eb deploy
```

## API Endpoints

- `POST /api/match` - JD-Resume matching
- `POST /api/generate-qa` - Generate interview questions
- `POST /api/analyze-call` - Analyze interview transcripts
- `POST /api/voice-stream` - TTS audio generation

## File Support

- PDF, DOCX, TXT resume formats
- Real-time voice transcription
- JSON structured responses