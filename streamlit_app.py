import streamlit as st
from groq import Groq
import tempfile
import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import json
from datetime import datetime
import pandas as pd
from collections import Counter
import re
import nltk
from nltk.tokenize import sent_tokenize
from duckduckgo_search import DDGS

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config for a wider layout and custom title with new theming options
st.set_page_config(
    page_title="JR Study Assist",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",  # Start with collapsed sidebar on mobile
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Your AI study companion - Mobile friendly!"
    }
)

# New theme configuration
st.set_page_config._theme = {
    "primaryColor": "#4CAF50",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F0F2F6",
    "textColor": "#262730",
    "font": "sans serif",
    "base": "light",
    "radius": "md",
}

# Custom CSS with updated styling including mobile optimizations
st.markdown("""
    <style>
    /* Modern styling with rounded corners */
    .stApp {
        max-width: 100%;
    }
    
    /* Mobile-responsive containers */
    @media screen and (max-width: 768px) {
        .stApp {
            padding: 0.5rem;
        }
        
        .styled-metric {
            padding: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        .chat-message {
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
        }
        
        /* Improve button spacing on mobile */
        .stButton > button {
            width: 100%;
            margin: 0.25rem 0;
        }
        
        /* Better radio button layout on mobile */
        .stRadio > div {
            flex-direction: column !important;
        }
        
        /* Adjust quiz options for mobile */
        div[role="radiogroup"] {
            flex-direction: column !important;
            gap: 0.5rem !important;
        }
        
        /* Make dataframes scrollable on mobile */
        .stDataFrame {
            overflow-x: auto;
        }
    }
    
    /* Base styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: var(--st-color-secondary-background);
        padding: 0.5rem;
        border-radius: 0.5rem;
        flex-wrap: wrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: var(--st-color-white);
        border-radius: 0.5rem;
        border: none;
        color: var(--st-color-primary);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--st-color-primary);
        color: var(--st-color-white);
    }
    
    .styled-metric {
        padding: 1.5rem;
        border-radius: 0.8rem;
        background: var(--st-color-white);
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .styled-metric:hover {
        transform: translateY(-2px);
    }
    
    div[data-testid="stFileUploader"] {
        padding: 1rem;
        border: 2px dashed var(--st-color-primary);
        border-radius: 1rem;
        background: var(--st-color-secondary-background);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--st-color-primary);
        background: rgba(76, 175, 80, 0.05);
    }
    
    .quiz-question {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 1rem;
        background: var(--st-color-white);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid var(--st-color-secondary-background);
    }
    
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 1rem;
        background: var(--st-color-secondary-background);
        border-left: 4px solid var(--st-color-primary);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    /* Badge styling with mobile optimization */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
        white-space: nowrap;
    }
    
    .badge-success {
        background: #4CAF50;
        color: white;
    }
    
    .badge-error {
        background: #F44336;
        color: white;
    }
    
    .badge-info {
        background: #2196F3;
        color: white;
    }
    
    .badge-warning {
        background: #FFC107;
        color: black;
    }
    
    /* Mobile-friendly version info */
    .version-info {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
        justify-content: flex-start;
    }
    
    /* Improve form controls on mobile */
    .stSelectbox, .stTextInput {
        max-width: 100% !important;
    }
    
    /* Better spacing for mobile buttons */
    .stButton {
        margin: 0.25rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for chat and documents
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "show_quiz_answers" not in st.session_state:
    st.session_state.show_quiz_answers = False
if "show_debug_info" not in st.session_state:
    st.session_state.show_debug_info = False

# Initialize session state for chat personality
if "chat_personality" not in st.session_state:
    st.session_state.chat_personality = "adaptive"  # Can be 'casual', 'formal', or 'adaptive'
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

def is_scanned_pdf(pdf_path):
    """Check if a PDF is scanned (image-based) or text-based."""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            return False  # Has text
    return True  # Likely scanned

def extract_text_from_pdf(pdf_path, use_ocr=False):
    """Extract text from PDF using either direct extraction or OCR."""
    if use_ocr:
        # Convert PDF to images and run OCR
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text
    else:
        # Direct text extraction
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def get_document_statistics(text):
    """Generate statistics about the document."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    word_freq = Counter(words)
    
    stats = {
        "Word Count": len(words),
        "Character Count": len(text),
        "Sentence Count": len(sentences),
        "Average Word Length": sum(len(word) for word in words) / len(words) if words else 0,
        "Most Common Words": dict(word_freq.most_common(5))
    }
    return stats

def extract_key_points(client, text):
    """Extract key points from the document using Groq."""
    messages = [
        {
            "role": "system",
            "content": "Extract 3-5 key points from the given text. Be concise and focus on the main ideas.",
        },
        {
            "role": "user",
            "content": f"Text:\n{text}\n\nExtract key points:",
        }
    ]
    
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        stream=False,
    )
    return response.choices[0].message.content

def chunk_text(text, max_chunk_size=2000):
    """Split text into smaller chunks based on sentences."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_quiz_questions(client, text_chunk, num_questions):
    """Generate quiz questions using smart model selection."""
    # Select appropriate model for quiz generation
    selected_model = get_best_available_model(client, question_type="analysis")
    
    messages = [
        {
            "role": "system",
            "content": """Generate multiple choice quiz questions based on the given text. 
            Each question should have 4 options (A, B, C, D) with exactly one correct answer.
            Format as JSON array with objects containing 'question', 'options' (array), and 'correct_answer' (0-3).
            Example: [{"question": "What is...?", "options": ["A", "B", "C", "D"], "correct_answer": 0}]""",
        },
        {
            "role": "user",
            "content": f"Text:\n{text_chunk}\n\nGenerate {num_questions} multiple choice questions:",
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            stream=False,
        )
        
        content = response.choices[0].message.content
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        return json.loads(content)
    except Exception as e:
        st.warning(f"Error generating questions: {str(e)}")
        return []

def validate_quiz_questions(questions):
    """Validate and filter quiz questions."""
    valid_questions = []
    for q in questions:
        try:
            if (isinstance(q, dict) and 
                'question' in q and 
                'options' in q and 
                'correct_answer' in q and 
                isinstance(q['options'], list) and 
                len(q['options']) == 4 and 
                isinstance(q['correct_answer'], int) and 
                0 <= q['correct_answer'] <= 3):
                valid_questions.append(q)
        except:
            continue
    return valid_questions

def generate_quiz(client, text, num_questions=5):
    """Generate a quiz using multiple models and text chunking."""
    try:
        st.info("🎯 Analyzing document and generating quiz...")
        
        # Split text into manageable chunks
        chunks = chunk_text(text)
        questions_per_chunk = max(1, num_questions // len(chunks))
        all_questions = []
        
        # Use progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            progress_text.text(f"Generating questions from section {i+1}/{len(chunks)}...")
            
            # Generate questions using smart model selection
            chunk_questions = generate_quiz_questions(
                client, 
                chunk, 
                questions_per_chunk
            )
            
            valid_questions = validate_quiz_questions(chunk_questions)
            all_questions.extend(valid_questions)
            
            # Update progress
            progress_bar.progress((i + 1) / len(chunks))
        
        progress_bar.empty()
        progress_text.empty()
        
        # Ensure we have enough questions
        if len(all_questions) < num_questions:
            st.warning(f"Could only generate {len(all_questions)} valid questions.")
        elif len(all_questions) > num_questions:
            all_questions = all_questions[:num_questions]
        
        if all_questions:
            st.success("✅ Quiz generated successfully!")
            return all_questions
        else:
            st.error("Failed to generate quiz questions. Please try again.")
            return None
            
    except Exception as e:
        st.error(f"Failed to generate quiz. Error: {str(e)}")
        return None

def display_quiz(quiz_data, show_answers=False):
    """Display quiz with enhanced styling and clear text formatting."""
    if not quiz_data:
        st.warning("No quiz data available. Please generate a quiz first.")
        return
    
    st.markdown("## Quiz Questions")
    
    for i, q in enumerate(quiz_data):
        with st.container():
            st.markdown(f"""### Question {i + 1}""")
            st.markdown(f"""{q["question"]}""")
            
            # Display options with corrected formatting
            options = [re.sub(r'^[A-D]\.\s*', '', opt.strip()) for opt in q["options"]]
            
            # Create columns for the question and potential answer feedback
            question_col, feedback_col = st.columns([3, 1])
            
            with question_col:
                for j, option in enumerate(options):
                    st.markdown(f"{chr(65+j)}. {option}")
                
                selected = st.radio(
                    "Select your answer:",
                    options=options,
                    key=f"q_{i}",
                    index=None if i not in st.session_state.quiz_answers else st.session_state.quiz_answers[i],
                    horizontal=True
                )
            
            if selected:
                selected_idx = options.index(selected)
                st.session_state.quiz_answers[i] = selected_idx
                
                with feedback_col:
                    if show_answers:
                        correct_answer = options[q["correct_answer"]]
                        if selected_idx == q["correct_answer"]:
                            st.markdown('<span class="badge badge-success">✅ Correct!</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f'<span class="badge badge-error">❌ Incorrect</span><br><br>'
                                f'<span class="badge badge-info">Correct: {chr(65+q["correct_answer"])}</span>', 
                                unsafe_allow_html=True
                            )
            
            st.markdown("---")
    
    # Show results in a modern card layout
    if st.session_state.quiz_answers:
        total_questions = len(quiz_data)
        answered_questions = len(st.session_state.quiz_answers)
        correct_answers = sum(1 for i in st.session_state.quiz_answers 
                            if st.session_state.quiz_answers[i] == quiz_data[i]["correct_answer"])
        score = (correct_answers/total_questions*100)
        
        st.markdown("## Quiz Results")
        
        # Create a results dashboard
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""
                <div class="styled-metric">
                    <h3>📊 Total Questions</h3>
                    <h2>{total_questions}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"""
                <div class="styled-metric">
                    <h3>✅ Correct Answers</h3>
                    <h2>{correct_answers}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown(f"""
                <div class="styled-metric">
                    <h3>📝 Answered</h3>
                    <h2>{answered_questions}/{total_questions}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[3]:
            score_color = "success" if score >= 70 else "warning" if score >= 50 else "error"
            st.markdown(f"""
                <div class="styled-metric">
                    <h3>🎯 Score</h3>
                    <h2><span class="badge badge-{score_color}">{score:.1f}%</span></h2>
                </div>
            """, unsafe_allow_html=True)

# Add personality settings to sidebar
def create_sidebar():
    """Create a well-organized sidebar with icons and sections."""
    with st.sidebar:
        st.title("🎯 Navigation")
        
        # Move file upload to sidebar
        uploaded_files = st.file_uploader(
            "📁 Upload Documents", 
            type=("pdf", "txt", "md"),
            accept_multiple_files=True,
            help="Support for PDF, TXT, and MD files"
        )

        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.documents:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        try:
                            if uploaded_file.type == "application/pdf":
                                if is_scanned_pdf(tmp_file_path):
                                    st.warning(f"📸 {uploaded_file.name} is scanned. Using OCR...")
                                    document = extract_text_from_pdf(tmp_file_path, use_ocr=True)
                                else:
                                    st.success(f"📑 Processing {uploaded_file.name}...")
                                    document = extract_text_from_pdf(tmp_file_path, use_ocr=False)
                            else:
                                document = uploaded_file.read().decode()

                            st.session_state.documents[uploaded_file.name] = document
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)

        # Document selector
        if st.session_state.documents:
            st.markdown("### 📑 Active Documents")
            selected_docs = st.multiselect(
                "Choose documents to analyze:",
                list(st.session_state.documents.keys()),
                default=list(st.session_state.documents.keys())[0]
            )
            
            if selected_docs:
                combined_text = "\n\n---\n\n".join([
                    f"Document: {doc}\n{st.session_state.documents[doc]}"
                    for doc in selected_docs
                ])
                st.session_state.document_text = combined_text

        st.markdown("---")
        
        # Initialize chat settings
        if "enable_web_search" not in st.session_state:
            st.session_state.enable_web_search = True
        if "show_recommendations" not in st.session_state:
            st.session_state.show_recommendations = True
            
        st.subheader("⚙️ Settings")
        st.session_state.chat_personality = st.selectbox(
            "🎭 Chat Style",
            options=["Adaptive", "Casual", "Formal"],
            help="Choose how I should talk with you"
        ).lower()
        
        st.session_state.enable_web_search = st.toggle(
            "🔍 Enable Web Search",
            value=st.session_state.enable_web_search,
            help="Enhance answers with web search results"
        )
        
        st.session_state.show_recommendations = st.toggle(
            "📚 Show Learning Resources",
            value=st.session_state.show_recommendations,
            help="Include related learning resources"
        )
        
        st.session_state.show_debug_info = st.toggle(
            "🛠️ Show Model Info",
            value=st.session_state.show_debug_info,
            help="Display which AI model is being used"
        )
        
        # Tools section
        if st.session_state.document_text:
            st.markdown("---")
            st.subheader("🛠️ Tools")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Statistics", use_container_width=True):
                    show_document_statistics()
            with col2:
                if st.button("📝 Key Points", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        key_points = extract_key_points(client, st.session_state.document_text)
                        st.info("Key Points:\n" + key_points)
            
            if st.button("🎯 Generate Quiz", use_container_width=True):
                with st.spinner("Creating quiz..."):
                    st.session_state.current_quiz = generate_quiz(
                        client, 
                        st.session_state.document_text,
                        num_questions=5
                    )
                    if st.session_state.current_quiz:
                        st.session_state.quiz_answers = {}
                        st.session_state.show_quiz_answers = False
                        st.rerun()

# Add welcome message function
def show_welcome_message():
    """Display a friendly welcome message based on personality setting."""
    if not st.session_state.welcome_shown and not st.session_state.messages:
        welcome_messages = {
            "casual": "Hey there! 👋 I'm your AI study buddy! Drop your documents here and let's chat about them! Feel free to ask me anything - I'm here to help make learning fun and easy! 😊",
            "formal": "Welcome to JR Study Assist. I am your dedicated AI learning assistant, ready to help you analyze documents and enhance your understanding through thoughtful discussion and analysis.",
            "adaptive": "Hello! 👋 I'm your AI study assistant, and I'll adapt to your style as we chat. Whether you prefer casual conversation or formal discussion, I'm here to help you learn and understand your documents better."
        }
        with st.chat_message("assistant"):
            st.markdown(f'<div class="chat-message">{welcome_messages[st.session_state.chat_personality]}</div>', unsafe_allow_html=True)
            st.session_state.welcome_shown = True

def show_document_statistics():
    """Display document statistics in a modern card layout."""
    stats = get_document_statistics(st.session_state.document_text)
    
    cols = st.columns(4)
    metrics = [
        ("Word Count", stats["Word Count"], "📝"),
        ("Character Count", stats["Character Count"], "📊"),
        ("Sentence Count", stats["Sentence Count"], "📈"),
        ("Avg Word Length", f"{stats['Average Word Length']:.1f}", "📏")
    ]
    
    for col, (label, value, icon) in zip(cols, metrics):
        with col:
            st.markdown(f'''
                <div class="styled-metric">
                    <h3>{icon} {label}</h3>
                    <h2>{value}</h2>
                </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("### Most Common Words")
    st.dataframe(
        pd.DataFrame(list(stats["Most Common Words"].items()), 
        columns=["Word", "Count"]),
        use_container_width=True,
        hide_index=True
    )

def export_section():
    """Handle all export functionality."""
    if st.button("📥 Export Chat History", use_container_width=True):
        chat_export = {
            "timestamp": datetime.now().isoformat(),
            "document_name": "Uploaded Document",
            "conversation": st.session_state.messages
        }
        st.download_button(
            "💾 Download Chat History (JSON)",
            data=json.dumps(chat_export, indent=2),
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True
        )

    if st.session_state.current_quiz:
        st.markdown("---")
        quiz_export = {
            "timestamp": datetime.now().isoformat(),
            "document_name": "Uploaded Document",
            "quiz": st.session_state.current_quiz,
            "user_answers": st.session_state.quiz_answers
        }
        st.download_button(
            "💾 Download Quiz Results (JSON)",
            data=json.dumps(quiz_export, indent=2),
            file_name="quiz_export.json",
            mime="application/json",
            use_container_width=True
        )

def web_search(query, max_results=3):
    """Perform web search and return relevant results."""
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                try:
                    results.append({
                        'title': r.get('title', 'No title'),
                        'link': r.get('url', ''),  # Changed from 'link' to 'url'
                        'snippet': r.get('body', '')
                    })
                except (AttributeError, KeyError):
                    continue
            return results
    except Exception as e:
        # Instead of showing error, quietly log it and return empty results
        print(f"Search error: {str(e)}")  # For debugging
        return []

def enhance_context_with_search(question, document_text):
    """Enhance context with relevant web search results."""
    # Only perform search if web search is enabled
    if st.session_state.enable_web_search:
        search_results = web_search(question)
        topic_search = web_search(f"learn more about {question}", max_results=2)
    else:
        search_results = []
        topic_search = []
    
    enhanced_context = {
        'document': document_text,
        'web_results': search_results,
        'recommendations': [
            {'title': r['title'], 'link': r['link']} 
            for r in topic_search if 'title' in r and 'link' in r
        ]
    }
    
    return enhanced_context

def format_response(content, context):
    """Format the response with citations and recommendations."""
    response = content

    if context.get('recommendations'):
        response += "\n\n📚 Learn more:\n"
        for rec in context['recommendations']:
            response += f"- [{rec['title']}]({rec['link']})\n"
    
    return response

def get_chat_model(question_type="general"):
    """
    Smart model selection for chat based on question type and conversation style.
    Returns tuple of (model_name, fallback_model)
    """
    models = {
        "general": {
            "primary": "llama-3.3-70b-versatile",  # Best for open-ended conversation
            "fallback": "meta-llama/llama-4-scout-17b-16e-instruct"  # Good at following instructions
        },
        "analysis": {
            "primary": "meta-llama/llama-4-scout-17b-16e-instruct",  # Better for analytical tasks
            "fallback": "llama-3.3-70b-versatile"
        }
    }
    
    # Enhanced question type detection
    analysis_keywords = ["analyze", "explain", "compare", "why", "how", "what if", "difference", "similarities"]
    casual_keywords = ["chat", "talk", "discuss", "tell me", "what do you think", "opinion", "feel"]
    
    if any(keyword in question_type.lower() for keyword in analysis_keywords):
        model_pair = models["analysis"]
    elif any(keyword in question_type.lower() for keyword in casual_keywords):
        model_pair = models["general"]
    else:
        # Default to versatile model for general conversation
        model_pair = models["general"]
    
    return model_pair["primary"], model_pair["fallback"]

def get_conversation_style(messages):
    """Determine the conversation style based on chat history."""
    if not messages:
        return "neutral"
        
    # Analyze last few messages for style
    recent_messages = messages[-3:]
    casual_markers = ["hey", "hi", "hello", "thanks", "thank you", "appreciate", "cool", "awesome"]
    formal_markers = ["could you please", "would you", "I would like", "kindly", "specifically"]
    
    casual_count = sum(1 for msg in recent_messages 
                      for marker in casual_markers 
                      if marker in msg.get("content", "").lower())
    formal_count = sum(1 for msg in recent_messages 
                      for marker in formal_markers 
                      if marker in msg.get("content", "").lower())
    
    if casual_count > formal_count:
        return "casual"
    elif formal_count > casual_count:
        return "formal"
    return "neutral"

def get_system_prompt(conv_style="neutral", has_documents=True):
    """Generate appropriate system prompt based on conversation style."""
    base_prompts = {
        "casual": """You are a friendly and helpful AI study assistant who loves to chat! 
        Think of yourself as a knowledgeable friend who's excited to help with learning.
        Be conversational, encouraging, and personable while staying informative.
        Feel free to use casual language, emoticons, and show enthusiasm, but always maintain accuracy.
        
        When responding:
        - Be warm and engaging while being informative
        - Use a mix of casual and professional language
        - Show excitement about the topics discussed
        - Offer encouragement and positive reinforcement
        - Feel free to ask clarifying questions
        - Share interesting related facts when relevant""",
        
        "formal": """You are a professional AI study assistant focused on providing precise and comprehensive assistance.
        Maintain a polite and professional tone while delivering thorough, well-structured responses.
        
        When responding:
        - Be clear, concise, and academically oriented
        - Use proper terminology and formal language
        - Provide well-structured, detailed explanations
        - Cite sources and references when applicable
        - Maintain professional courtesy
        - Focus on accuracy and precision""",
        
        "neutral": """You are a versatile AI study assistant who adapts to the user's style.
        Balance friendliness with professionalism while providing accurate and helpful information.
        
        When responding:
        - Be natural and adaptive in conversation
        - Match the user's level of formality
        - Focus on being helpful and informative
        - Provide clear and accurate information
        - Be encouraging and supportive
        - Maintain a good balance of detail and clarity"""
    }
    
    # Add document-specific guidelines
    doc_guidelines = """
    Guidelines for working with documents:
    1. Focus primarily on document content for answers
    2. Use web search results to enhance understanding when relevant
    3. Suggest related learning resources when helpful
    4. Distinguish between document content and external information
    5. Format responses clearly with markdown
    6. Cite sources appropriately
    7. Be proactive in suggesting relevant connections
    8. Offer examples when they help clarify concepts
    """
    
    prompt = base_prompts[conv_style]
    if has_documents:
        prompt += "\n\n" + doc_guidelines
        
    return prompt

def is_model_available(client, model_name):
    """Check if a model is available and responding."""
    try:
        # Try a simple completion to test model availability
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception:
        return False

def get_best_available_model(client, question_type="general"):
    """Get the best available model for the current chat interaction."""
    primary_model, fallback_model = get_chat_model(question_type)
    
    # Try primary model first
    if is_model_available(client, primary_model):
        return primary_model
    
    # Fall back to secondary model if primary is unavailable
    if is_model_available(client, fallback_model):
        return fallback_model
    
    # If both specific models are unavailable, use a stable fallback
    return "llama3-70b-8192"

# Create Groq client using the API key from secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Main app layout modification
def main():
    st.title("📚 JR Study Assist")
    
    # Version info in a container
    with st.container():
        st.markdown("""
            <div class="version-info">
                <div>Your AI-powered study companion for document analysis and learning</div>
                <span class="badge badge-info">v1.44.2</span>
                <span class="badge badge-success">Made with Streamlit</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create the sidebar
    create_sidebar()
    
    # Show welcome message
    show_welcome_message()
    
    # Chat interface
    st.markdown("### 💬 Chat Interface")
    chat_container = st.container()
    
    with chat_container:
        # Display message history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(f'''
                    <div class="chat-message">
                        {message["content"]}
                    </div>
                ''', unsafe_allow_html=True)

        # Chat input
        if question := st.chat_input(
            "Ask a question about your documents..." if st.session_state.document_text 
            else "👋 Hi! Upload documents in the sidebar to get started!"
        ):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-message">{question}</div>', unsafe_allow_html=True)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..." if st.session_state.chat_personality == "casual" else "Processing..."):
                    # Get conversation style
                    conv_style = st.session_state.chat_personality if st.session_state.chat_personality != "adaptive" else get_conversation_style(st.session_state.messages)
                    
                    # Check if the question is about document analysis or quiz
                    doc_keywords = ["analyze document", "read document", "summarize document", "generate quiz", "create quiz"]
                    if any(keyword in question.lower() for keyword in doc_keywords) and not st.session_state.document_text:
                        response = "Please upload a document in the navigation bar (👈 left sidebar) first to use document analysis features!"
                    else:
                        # Regular chat mode
                        system_prompt = f"""You are a friendly and helpful AI assistant who can engage in natural conversation
                        while also helping with document analysis and study when requested. Keep responses clear and engaging.
                        Conversation style: {conv_style}"""
                        
                        selected_model = get_best_available_model(client, "general")
                        
                        messages = [
                            {"role": "system", "content": system_prompt}
                        ] + st.session_state.messages[-5:] + [
                            {"role": "user", "content": question}
                        ]

                        stream = client.chat.completions.create(
                            model=selected_model,
                            messages=messages,
                            stream=True,
                        )

                        response = ""
                        message_placeholder = st.empty()
                        
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                response += content
                                message_placeholder.markdown(
                                    f'<div class="chat-message">{response}</div>', 
                                    unsafe_allow_html=True
                                )

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
    
    # Show quiz if it exists
    if st.session_state.current_quiz:
        st.markdown("### 📝 Quiz")
        display_quiz(st.session_state.current_quiz, st.session_state.show_quiz_answers)

if __name__ == "__main__":
    main()