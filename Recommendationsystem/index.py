from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template import RequestContext
import pymysql
from datetime import date
from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.conf import settings
from django.contrib import messages
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from application.models import myuser
from application.models import question
from application.models import answer
import random
from gtts import gTTS
import os
import spacy
import nltk
import multiprocessing
from playsound import playsound
import platform

try:
    mydb = pymysql.connect(host="localhost", user="root", password="Tanishq#22", database="ai_interview")
except pymysql.Error as e:
    print(f"Database connection failed: {e}")
    mydb = None

def is_db_connected():
    """Check if database is connected and working"""
    if mydb is None:
        return False
    try:
        mydb.ping(reconnect=True)
        return True
    except:
        return False

try:
    nltk.download('punkt')
    print("Loading spaCy model...")
    # Try to load the large model first (best accuracy)
    try:
        nlp = spacy.load("en_core_web_lg")
        print(f"Large model loaded successfully: {nlp.meta['name']}")
        print(f"Has word vectors: {nlp.vocab.vectors.shape}")
    except Exception as lg_error:
        print(f"Failed to load large model: {lg_error}")
        # Try medium model as fallback
        try:
            nlp = spacy.load("en_core_web_md")
            print(f"Medium model loaded successfully: {nlp.meta['name']}")
            print(f"Has word vectors: {nlp.vocab.vectors.shape}")
        except Exception as md_error:
            print(f"Failed to load medium model: {md_error}")
            # Try small model as last resort
            try:
                nlp = spacy.load("en_core_web_sm")
                print(f"Small model loaded successfully: {nlp.meta['name']}")
                print("Note: Small model has limited word vectors - similarity scores may be less accurate")
            except Exception as sm_error:
                print(f"Failed to load small model: {sm_error}")
                print("Trying to install spaCy models...")
                try:
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    nlp = spacy.load("en_core_web_sm")
                    print(f"Small model installed and loaded: {nlp.meta['name']}")
                except Exception as install_error:
                    print(f"Failed to install spaCy models: {install_error}")
                    nlp = None
except Exception as e:
    print(f"Critical error loading NLP models: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
    nlp = None

def convert_text_to_speech(text):
    tts = gTTS(text)
    # Save to a relative path that works on all platforms
    audio_path = "static/question.mp3"
    tts.save(audio_path)
    # Use platform-appropriate command to play audio
    if platform.system() == "Darwin":  # macOS
        os.system(f"open {audio_path}")
    elif platform.system() == "Windows":
        os.system(f"start {audio_path}")
    else:  # Linux
        os.system(f"xdg-open {audio_path}")
    #playsound(audio_path)
    #os.remove(audio_path)

# Global variables removed - using session storage instead

def calculate_hybrid_similarity(user_answer, correct_answer):
    """
    Calculate similarity using strict semantic correctness scoring
    """
    try:
        # Special case for "No answer provided"
        if user_answer.lower().strip() == "no answer provided":
            return 0.0
        
        # 1. Basic spaCy similarity (very low weight)
        user_doc = nlp(user_answer.lower().strip())
        correct_doc = nlp(correct_answer.lower().strip())
        spacy_sim = user_doc.similarity(correct_doc) * 100
        
        # 2. Strict keyword matching (higher weight, exact matches only)
        user_words = set([token.lemma_.lower() for token in user_doc if not token.is_stop and len(token.text) > 2])
        correct_words = set([token.lemma_.lower() for token in correct_doc if not token.is_stop and len(token.text) > 2])
        
        if len(correct_words) == 0:
            keyword_score = 0
        else:
            # Only count exact word matches, not partial matches
            exact_matches = user_words.intersection(correct_words)
            keyword_score = (len(exact_matches) / len(correct_words)) * 100
        
        # 3. Answer quality check (strict penalties)
        quality_score = 100
        if len(user_answer.strip()) < 5:
            quality_score = 20  # Heavy penalty for very short answers
        elif len(user_answer.strip()) < 10:
            quality_score = 40  # Heavy penalty for short answers
        elif len(user_answer.strip()) < 20:
            quality_score = 70  # Moderate penalty for medium answers
        
        # 4. Semantic correctness check (very strict)
        if spacy_sim < 15:  # Very low similarity
            semantic_penalty = 0.2  # Reduce score by 80%
        elif spacy_sim < 25:  # Low similarity
            semantic_penalty = 0.4  # Reduce score by 60%
        elif spacy_sim < 35:  # Medium-low similarity
            semantic_penalty = 0.6  # Reduce score by 40%
        else:
            semantic_penalty = 1.0  # No penalty
        
        # 5. Content relevance check (penalize completely wrong topics)
        content_penalty = 1.0
        if "snake" in user_answer.lower() and "python" in correct_answer.lower():
            content_penalty = 0.1  # 90% penalty for snake references to Python programming
        elif "programming" not in user_answer.lower() and "programming" in correct_answer.lower():
            content_penalty = 0.3  # 70% penalty for missing programming context
        
        # 6. Calculate final weighted score (much stricter)
        final_score = (
            (spacy_sim * 0.15) +          # Only 15% weight to spaCy similarity
            (keyword_score * 0.55) +       # 55% weight to exact keyword matching
            (quality_score * 0.25) +       # 25% weight to answer quality
            (0 * 0.05)                     # 5% reserved
        ) * semantic_penalty * content_penalty
        
        # Ensure score is between 0 and 100
        final_score = max(0, min(100, final_score))
        
        # Debug output
        print(f"  Debug - spaCy: {spacy_sim:.2f}, Keywords: {keyword_score:.2f}, Quality: {quality_score:.2f}")
        print(f"  Debug - Semantic penalty: {semantic_penalty}, Content penalty: {content_penalty}")
        print(f"  Debug - Final score: {final_score:.2f}")
        
        return round(final_score, 2)
        
    except Exception as e:
        print(f"Error in hybrid similarity calculation: {e}")
        # Fallback to basic similarity if hybrid fails
        try:
            user_doc = nlp(user_answer.lower().strip())
            correct_doc = nlp(correct_answer.lower().strip())
            return round(user_doc.similarity(correct_doc) * 100, 2)
        except:
            return 0.0


def chatbot1(request):
    import pandas as pd
    import speech_recognition as sr
    
    # Check if user is logged in
    if 'uid' not in request.session:
        messages.error(request, "Please log in to submit answers.")
        return redirect('login')
    
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return redirect('login')
    
    if nlp is None:
        messages.error(request, "NLP models not loaded. Please try again later.")
        return redirect('login')
    
    # Get questions and answers from session
    questions = request.session.get('questions', [])
    answers = request.session.get('answers', [])
    current_index = request.session.get('current_question_index', 0)
    
    # Validate session data integrity
    if not questions or not answers:
        messages.error(request, "No questions loaded. Please select a subject first.")
        return redirect('chatbot')
    
    if current_index >= len(questions) or current_index >= len(answers):
        messages.error(request, "Question index out of range. Please restart the quiz.")
        # Clear corrupted session data
        request.session.pop('questions', None)
        request.session.pop('answers', None)
        request.session.pop('current_question_index', None)
        request.session.pop('subject', None)
        return redirect('chatbot')
    
    que = request.POST.get('q6')
    user_answer = request.POST.get('answer', '').strip()
    
    if not que:
        messages.error(request, "Missing question")
        return redirect('login')
    
    # Handle empty answers by setting a default value
    if not user_answer:
        user_answer = "No answer provided"
    
    print("Question ", que)
    print("Answer ", user_answer)
    
    try:
        canswer = answers[current_index]
    except IndexError:
        messages.error(request, "Question index out of range")
        return redirect('login')
    
    try:
        # Use hybrid scoring for more accurate results
        similarity = calculate_hybrid_similarity(user_answer, canswer)
        print(f"Hybrid similarity score: {similarity}")
        print(f"User answer: '{user_answer}'")
        print(f"Correct answer: '{canswer}'")
        
        c1 = mydb.cursor()
        q = "insert into answers(question,ans,score)values(%s,%s,%s)"
        
        c1.execute(q, (que, user_answer, str(similarity)))
        mydb.commit()
    except Exception as e:
        messages.error(request, f"Error processing answer: {str(e)}")
        return redirect('login')
   
    # Move to next question
    current_index += 1
    request.session['current_question_index'] = current_index
    
    # Check if we've completed all questions
    if current_index >= len(questions):
        # All questions completed - clear session data and show results
        request.session.pop('questions', None)
        request.session.pop('answers', None)
        request.session.pop('current_question_index', None)
        request.session.pop('subject', None)
        
        messages.success(request, "All questions completed! Here are your results:")
        return question_display(request)
    else:
        # Load next question
        try:
            cquestion = questions[current_index]
            convert_text_to_speech(cquestion)
            return render(request, "question.html", {'question': cquestion, 'count': current_index, 'uid': request.session.get('uid')})
        except IndexError:
            # Handle case where current_index is out of bounds
            messages.error(request, "Question index out of range. Please restart the quiz.")
            # Clear corrupted session data
            request.session.pop('questions', None)
            request.session.pop('answers', None)
            request.session.pop('current_question_index', None)
            request.session.pop('subject', None)
            return redirect('chatbot')
        except Exception as e:
            messages.error(request, f"Error loading next question: {str(e)}")
            return redirect('login')

def c_question_display(request):
    # Check if user is logged in
    if 'uid' not in request.session:
        messages.error(request, "Please log in to access questions.")
        return redirect('login')
        
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return redirect('login')
        
    try:
        # Clear previous answers
        c1 = mydb.cursor()
        c1.execute("delete from answers")
        
        # Get the subject from request
        data = request.GET.get('val')
        if not data:
            messages.error(request, "No subject specified")
            return redirect('chatbot')
        
        # Query for questions - try exact match first
        q1 = "select * from questions where subject=%s"
        cur = mydb.cursor()
        cur.execute(q1, (data,))
        res = cur.fetchall()
        
        # If no exact match, try case-insensitive search
        if not res:
            q1 = "select * from questions where LOWER(subject)=LOWER(%s)"
            cur.execute(q1, (data,))
            res = cur.fetchall()
        
        # If still no match, try trimming whitespace
        if not res:
            q1 = "select * from questions where TRIM(subject)=TRIM(%s)"
            cur.execute(q1, (data,))
            res = cur.fetchall()
        
        # Debug: Let's see what subjects are actually in the database
        if not res:
            debug_cur = mydb.cursor()
            debug_cur.execute("SELECT DISTINCT subject FROM questions")
            all_subjects = debug_cur.fetchall()
            print(f"Available subjects in database: {[s[0] for s in all_subjects]}")
            print(f"Looking for subject: '{data}'")
            debug_cur.close()
            
            messages.error(request, f"No questions found for '{data}'. Available subjects: {[s[0] for s in all_subjects]}. Please add some questions first.")
            return redirect('chatbot')
        
        # Store questions and answers in session
        questions = []
        answers = []
        for x in res:
            questions.append(x[1])  # question
            answers.append(x[3])    # answer
        
        # Store in session
        request.session['questions'] = questions
        request.session['answers'] = answers
        request.session['current_question_index'] = 0
        request.session['subject'] = data
        
        # Get first question
        cquestion = questions[0]
        
        # Convert to speech
        convert_text_to_speech(cquestion)
        
        print(f"Loaded {len(questions)} questions for {data}")
        return render(request, "question.html", {'question': cquestion, 'count': 0, 'uid': request.session.get('uid')})
        
    except Exception as e:
        messages.error(request, f"Error loading questions: {str(e)}")
        return redirect('chatbot')


def question_display(request):
    # Check if user is logged in
    if 'uid' not in request.session:
        messages.error(request, "Please log in to view results.")
        return redirect('login')
        
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return redirect('login')
        
    try:
        payload = []
        sql = "SELECT * FROM answers"
        c1 = mydb.cursor()
        c1.execute(sql)
        rows = c1.fetchall()
        
        if not rows:
            messages.info(request, "No answers found. Please complete a quiz first.")
            return redirect('chatbot')
            
        for x in rows:
            content = {'answers':x[2], "similarity":x[3], "que": x[1]}
            payload.append(content)
        return render(request, "click.html", {'list': {'items': payload}})
    except Exception as e:
        messages.error(request, f"Error loading answers: {str(e)}")
        return redirect('login')


def about(request):
    return render(request,"about.html")

def service(request):
    return render(request,"service.html")

def admindashboard(request):
     return render(request,"admindashboard.html")

def questions(request):
     return render(request,"inputquestions.html")

def inputquestions(request):
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return render(request, "inputquestions.html")
    
    # Handle POST request (form submission)
    if request.method == 'POST':
        inputquestions_text = request.POST.get("inputquestions")
        inputanswer = request.POST.get("inputanswer")
        programming_language = request.POST.get("programminglanguage")
        
        # Validate input
        if not inputquestions_text or not inputanswer or not programming_language:
            messages.error(request, "All fields are required!")
            return render(request, "inputquestions.html")
        
        try:
            sql = "insert into questions(question,subject,answer)values(%s,%s,%s)"
            c1 = mydb.cursor()
            c1.execute(sql, (inputquestions_text, programming_language, inputanswer))
            mydb.commit()
            messages.success(request, "Question added successfully!")
            return render(request, "inputquestions.html")
        except Exception as e:
            messages.error(request, f"Error adding question: {str(e)}")
            return render(request, "inputquestions.html")
    
    # Handle GET request (just viewing the page)
    return render(request, "inputquestions.html")

def showquestion(request):
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return render(request, "showquestion.html", {'list': {'items': []}})
        
    try:
        payload = []
        content = {}
        payload = []
        
        # Get subject filter from request
        subject_filter = request.GET.get('sub')
        
        if subject_filter:
            # Filter by specific subject
            q1 = "select * from questions where subject=%s"
            cur = mydb.cursor()
            cur.execute(q1, (subject_filter,))
        else:
            # Show all questions
            q1 = "select * from questions"
            cur = mydb.cursor()
            cur.execute(q1)
            
        res = cur.fetchall()
        for x in res:
            content = {'uid': x[0], 'que': x[1], 'subject': x[2], "answer": x[3]}
            payload.append(content)
            content = {}
        # Render data to template
        return render(request, "showquestion.html", {'list': {'items': payload}})
    except Exception as e:
        messages.error(request, f"Error loading questions: {str(e)}")
        return render(request, "showquestion.html", {'list': {'items': []}})

    
def dashboard(request):
    return render(request,"admindashboard.html")

    
def login(request):
    # Clear any existing messages to prevent them from showing on login page
    list(messages.get_messages(request))
    return render(request,"loginpanel.html")
    
def logout(request):
    # Clear any existing messages to prevent them from showing on logout page
    list(messages.get_messages(request))
    return render(request,"loginpanel.html")

def register(request):
    return render(request,"registrationPanel.html")

def dologin(request):
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return redirect('login')
        
    email = request.POST.get('email')
    password = request.POST.get('password')
    if email == "admin" and password == "admin":
        return redirect('dashboard')
    else:
        try:
            sql = "SELECT * FROM userdata"
            c1 = mydb.cursor()
            c1.execute(sql)
            rows = c1.fetchall()
            uid = ""
            img = ""
            tf = ""
            ispresent = False
            for x in rows:
                if(email == x[3] and password == x[4]):
                    name = x[1]
                    uid = x[0]
                    ispresent = True
            if (ispresent):
                request.session['uname'] = name
                request.session['uid'] = uid
                return redirect('UserDashboard')  # Redirect to the index page after successful login
            else:
                messages.error(request, "Invalid email or password. Please try again.")
                return redirect('login')
        except Exception as e:
            messages.error(request, "An error occurred during login. Please try again.")
            return redirect('login')


def prevpred(request):
    payload = []
    uid = request.session.get('uid')

    answers = answer.objects.filter(uid=uid)

    # Prepare payload
    for answer in answers:
        content = {'answers': answer.answers}
        payload.append(content)

    return render(request, "prevpred.html", {'list': {'items': payload}})


def doregister(request):
    if request.method == 'POST':
        if not is_db_connected():
            messages.error(request, "Database connection error. Please try again later.")
            return render(request, "registrationPanel.html")
            
        username = request.POST.get('username')
        contact = request.POST.get('contact')
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        # Validate required fields
        if not username or not contact or not email or not password:
            messages.error(request, "All fields are required. Please fill in all the information.")
            return render(request, "registrationPanel.html")
        
        try:
            sql = "insert into userdata(uname,contact,email,pass)values(%s,%s,%s,%s)"
            c1 = mydb.cursor()
            c1.execute(sql,(username,contact,email,password))
            mydb.commit()
            messages.success(request, "Registration successful! Please login.")
            return render(request, "loginpanel.html")
        except pymysql.IntegrityError as e:
            if "Duplicate entry" in str(e):
                messages.error(request, "Email already exists. Please use a different email address.")
            else:
                messages.error(request, f"Registration failed: {str(e)}")
            return render(request, "registrationPanel.html")
        except Exception as e:
            messages.error(request, f"Registration failed: {str(e)}")
            return render(request, "registrationPanel.html")
    else:
        return render(request, "registrationPanel.html") 

def viewpredicadmin(request):
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return render(request, "viewpredadmin.html", {'list': {'items': []}})
        
    try:
        content={}
        payload=[]
        q1="select * from smp"
        cur=mydb.cursor()
        cur.execute(q1)
        res=cur.fetchall()
        for x in res:
            content={'s1':x[0],"s2":x[1],"s3":x[2],"s4":x[3],'s5':x[4],"s6":x[5],"s7":x[6],"s8":x[7],"pred":x[8],"acc":x[9]}
            payload.append(content)
            content={}
        return render(request,"viewpredadmin.html",{'list': {'items':payload}})
    except Exception as e:
        messages.error(request, f"Error loading prediction data: {str(e)}")
        return render(request, "viewpredadmin.html", {'list': {'items': []}})


# Removed duplicate function definition

def myprofile(request):
    # Clear any existing messages to prevent them from showing on profile page
    list(messages.get_messages(request))
    
    # Initialize an empty payload list
    payload = []

    # Retrieve user ID from session (using 'uid' as set in dologin)
    uid = request.session.get('uid')
    
    if not uid:
        messages.error(request, "Please login to view your profile.")
        return redirect('login')
    
    try:
        # Get user data from the userdata table using PyMySQL
        if not is_db_connected():
            messages.error(request, "Database connection error. Please try again later.")
            return render(request, "myprofile.html", {'list': {'items': []}})
        
        sql = "SELECT * FROM userdata WHERE id = %s"
        c1 = mydb.cursor()
        c1.execute(sql, (uid,))
        rows = c1.fetchall()
        
        if rows:
            user_data = rows[0]
            content = {
                'name': user_data[1],  # uname
                'contact': user_data[2],  # contact
                'email': user_data[3]   # email
            }
            payload.append(content)
        
        return render(request, "myprofile.html", {'list': {'items': payload}})
    except Exception as e:
        messages.error(request, f"Error loading profile: {str(e)}")
        return render(request, "myprofile.html", {'list': {'items': []}})


def viewuser(request):
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return render(request, 'viewuserprofile.html', {'list': {'items': []}})
        
    try:
        sql = "SELECT * FROM  userdata"
        c1 = mydb.cursor()
        c1.execute(sql)
        rows = c1.fetchall()
        content={}
        payload=[]
        ispresent = False
        for x in rows:
            content={'id':x[0],'name':x[1],'contact':x[2],'email':x[3]}
            payload.append(content)
            content={}
        
        return render(request, 'viewuserprofile.html',{'list': {'items':payload}})
    except Exception as e:
        messages.error(request, "Error loading user data. Please try again later.")
        return render(request, 'viewuserprofile.html', {'list': {'items': []}})
    
def UserDashboard(request):
        return render(request,"UserDashboard.html")     

def livepred(request):
    return render(request,"predict.html")


def chatbot(request):
    # Check if user is logged in
    if 'uid' not in request.session:
        messages.error(request, "Please log in to access the chatbot.")
        return redirect('login')
    return render(request, 'chatbot.html')

def chat(request):
    # Check if user is logged in
    if 'uid' not in request.session:
        messages.error(request, "Please log in to access the chatbot.")
        return redirect('login')
    return render(request,"chatbot.html")

# Cleanup function to close database connection
import atexit

def cleanup():
    if mydb and mydb.open:
        mydb.close()

atexit.register(cleanup)

def delete_question(request):
    """Delete a question from the database"""
    if request.method != 'POST':
        messages.error(request, "Invalid request method")
        return redirect('showquestion')
    
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return redirect('showquestion')
    
    uid = request.POST.get('uid')
    if not uid:
        messages.error(request, "Question ID not provided")
        return redirect('showquestion')
    
    try:
        # First check if the question exists
        check_sql = "SELECT * FROM questions WHERE id = %s"
        cur = mydb.cursor()
        cur.execute(check_sql, (uid,))
        question_exists = cur.fetchone()
        
        if not question_exists:
            messages.error(request, "Question not found")
            return redirect('showquestion')
        
        # Delete the question
        delete_sql = "DELETE FROM questions WHERE id = %s"
        cur.execute(delete_sql, (uid,))
        mydb.commit()
        
        messages.success(request, "Question deleted successfully")
        
    except Exception as e:
        messages.error(request, f"Error deleting question: {str(e)}")
        mydb.rollback()
    
    finally:
        if cur:
            cur.close()
    
    return redirect('showquestion')


def delete_user(request):
    """Delete a user from the database"""
    if request.method != 'POST':
        messages.error(request, "Invalid request method")
        return redirect('viewuser')
    
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return redirect('viewuser')
    
    uid = request.POST.get('uid')
    if not uid:
        messages.error(request, "User ID not provided")
        return redirect('viewuser')
    
    try:
        # First check if the user exists
        check_sql = "SELECT * FROM userdata WHERE id = %s"
        cur = mydb.cursor()
        cur.execute(check_sql, (uid,))
        user_exists = cur.fetchone()
        
        if not user_exists:
            messages.error(request, "User not found")
            return redirect('viewuser')
        
        # Delete the user
        delete_sql = "DELETE FROM userdata WHERE id = %s"
        cur.execute(delete_sql, (uid,))
        mydb.commit()
        
        messages.success(request, "User deleted successfully")
        
    except Exception as e:
        messages.error(request, f"Error deleting user: {str(e)}")
        mydb.rollback()
    
    finally:
        if cur:
            cur.close()
    
    return redirect('viewuser')


def edit_profile(request):
    """Edit user profile information"""
    if request.method != 'POST':
        messages.error(request, "Invalid request method")
        return redirect('myprofile')
    
    # Check if user is logged in
    uid = request.session.get('uid')
    if not uid:
        messages.error(request, "Please login to edit your profile.")
        return redirect('login')
    
    if not is_db_connected():
        messages.error(request, "Database connection error. Please try again later.")
        return redirect('myprofile')
    
    try:
        # Get form data
        name = request.POST.get('name')
        contact = request.POST.get('contact')
        email = request.POST.get('email')
        
        # Basic validation
        if not name or not contact or not email:
            messages.error(request, "All fields are required")
            return redirect('myprofile')
        
        # Check if email is already taken by another user
        check_sql = "SELECT id FROM userdata WHERE email = %s AND id != %s"
        cur = mydb.cursor()
        cur.execute(check_sql, (email, uid))
        existing_user = cur.fetchone()
        
        if existing_user:
            messages.error(request, "Email address is already in use by another user")
            return redirect('myprofile')
        
        # Update the user profile
        update_sql = "UPDATE userdata SET uname = %s, contact = %s, email = %s WHERE id = %s"
        cur.execute(update_sql, (name, contact, email, uid))
        mydb.commit()
        
        messages.success(request, "Profile updated successfully!")
        
    except Exception as e:
        messages.error(request, f"Error updating profile: {str(e)}")
        mydb.rollback()
    
    finally:
        if cur:
            cur.close()
    
    return redirect('myprofile')
