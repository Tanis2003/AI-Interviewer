# 🤖 AI Interviewer System

An AI-powered Django application that conducts interviews using text-to-speech and evaluates answers with an NLP-based similarity score.

-----

## ✨ Features

  * **User Authentication:** Secure user registration and login system.
  * **Admin Panel:** An admin-only section to add, view, and delete interview questions and manage users.
  * **Dynamic Questioning:** Users can select a topic (e.g., Python, C++, Java), and the system will fetch relevant questions.
  * **Text-to-Speech:** Questions are read aloud for an immersive experience.
  * **AI-Powered Evaluation:** User answers are scored in real-time using an NLP-based similarity algorithm.
  * **Results Dashboard:** After the interview, users can see a list of their answers and the corresponding scores.
  * **Profile Management:** Users can view and edit their profile information.

-----

## ⚙️ Methodology

The core of this project lies in its ability to evaluate user-provided answers. This is not a simple keyword search; it uses a hybrid NLP model to determine the semantic correctness of a response.

  * **Hybrid Similarity Scoring:** The `calculate_hybrid_similarity` function evaluates answers using a weighted algorithm.

      * **Keyword Matching (55% weight):** It tokenizes both the user's answer and the correct answer, lemmatizes them, removes stop words, and calculates a score based on the intersection of exact keywords. This forms the backbone of the score.
      * **Answer Quality (25% weight):** It applies strict penalties for overly short answers, encouraging users to be descriptive.
      * **spaCy Semantic Similarity (15% weight):** It uses a pre-trained NLP model from spaCy (`en_core_web_lg`) to calculate the semantic similarity between the user's answer and the correct one. This helps capture the contextual meaning even if different words are used.
      * **Penalties:** The final score is adjusted downwards by *semantic* and *content* penalties if the answer is completely off-topic (e.g., talking about a snake for a Python programming question) or has very low semantic similarity.

  * **Google Text-to-Speech (gTTS) API:** To make the interview experience more interactive, the application uses the `gTTS` library. This library makes an API call to Google's Text-to-Speech service to convert the question text into an `.mp3` audio file, which is then played back to the user automatically.

-----

## 🚀 Local Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

  * Python 3.8+
  * pip (Python package installer)
  * A running MySQL server (e.g., via XAMPP, MAMP, or a standalone installation)
  * Git

### 1\. Clone the Repository

Open your terminal and clone the repository to your local machine.

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2\. Create a Virtual Environment

It's a best practice to use a virtual environment to keep project dependencies isolated.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 3\. Install Required Libraries

Install all the necessary Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4\. Download the NLP Model

The project uses a large spaCy model for the best semantic analysis. Download it using the following command:

```bash
python -m spacy download en_core_web_lg
```

### 5\. Configure the Database

This is the most critical part of the setup.

1.  **Start your MySQL server** and open a database management tool (like phpMyAdmin or MySQL Workbench).
2.  **Create a new database.** The project expects the database to be named `ai_interview`.
3.  **Update the database password in the code.** Open the `index.py` file and find the following line (around line 25):
    ```python
    # IMPORTANT: Change the password to your own MySQL root password!
    mydb = pymysql.connect(host="localhost", user="root", password="Tanishq#22", database="ai_interview")
    ```
4.  **Create the required tables.** Run the following SQL commands in your `ai_interview` database to set up the necessary tables:
    ```sql
    CREATE TABLE `userdata` (
      `id` int(11) NOT NULL AUTO_INCREMENT,
      `uname` varchar(255) NOT NULL,
      `contact` varchar(20) NOT NULL,
      `email` varchar(255) NOT NULL UNIQUE,
      `pass` varchar(255) NOT NULL,
      PRIMARY KEY (`id`)
    );

    CREATE TABLE `questions` (
      `id` int(11) NOT NULL AUTO_INCREMENT,
      `question` TEXT NOT NULL,
      `subject` varchar(255) NOT NULL,
      `answer` TEXT NOT NULL,
      PRIMARY KEY (`id`)
    );

    CREATE TABLE `answers` (
      `id` int(11) NOT NULL AUTO_INCREMENT,
      `question` TEXT,
      `ans` TEXT,
      `score` varchar(255),
      PRIMARY KEY (`id`)
    );
    ```

### 6\. Run the Django Server

You are now ready to run the project\!

```bash
python manage.py runserver
```

Open your web browser and go to **`http://127.0.0.1:8000/`**.

### Admin Credentials

To access the admin dashboard at `http://127.0.0.1:8000/dashboard/`, use the following hardcoded credentials:

  * **Username:** `admin`
  * **Password:** `admin`

-----

## 💻 Platform Notice

This project was developed and tested on **macOS**. The text-to-speech feature uses OS-specific commands to automatically play the generated audio file.

If you are running this on **Windows**, you may need to ensure that the file paths and system commands are compatible. The code in `index.py` already includes logic to handle different operating systems, but be aware of potential path differences.

```python
# In convert_text_to_speech function
if platform.system() == "Darwin":  # macOS
    os.system(f"open {audio_path}")
elif platform.system() == "Windows":
    os.system(f"start {audio_path}")
else:  # Linux
    os.system(f"xdg-open {audio_path}")

```
