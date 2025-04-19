# JR Study Assist

An AI-powered study companion application that helps users analyze documents, generate quizzes, and learn more effectively.

## Features

- 📚 Document Analysis: Upload and process PDF, TXT, and MD files
- 💡 Interactive Q&A: Chat with your documents using AI
- 📝 Smart Quiz Generation: Create customized quizzes from your documents
- 📊 Document Statistics: Get insights about your study materials
- 💾 Export Functionality: Save your chat history and quiz results

## Technical Features

- PDF text extraction with OCR support for scanned documents
- Multi-document analysis
- Interactive chat interface
- Customizable quiz generation
- Modern, responsive UI with Streamlit
- Document statistics and analytics

## How to run it

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

2. Install system dependencies (for PDF processing):
   - Ubuntu: `sudo apt install tesseract-ocr poppler-utils`
   - Windows: Install Tesseract and Poppler manually

3. Add your Groq API key to `.streamlit/secrets.toml`

4. Run the app:
   ```
   streamlit run streamlit_app.py
   ```
