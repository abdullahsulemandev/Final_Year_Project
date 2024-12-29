import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

DB_PATH = 'database.db'

# Initialize the database and create tables if they don't exist
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # Create users table
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                     )''')
        # Create feedback table
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        feedback_text TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                     )''')
        # Create filters table using user_email as a foreign key
        c.execute('''CREATE TABLE IF NOT EXISTS filters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_email TEXT NOT NULL,
                        filter TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        video_id TEXT NOT NULL,
                        FOREIGN KEY (user_email) REFERENCES users (email)
                    )''')
        conn.commit()

# Add a new user with hashed password to the users table
def add_user(email, password):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            hashed_password = generate_password_hash(password)
            c.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, hashed_password))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False  # User already exists
    except Exception as e:
        print(f"Error adding user: {e}")
        return False

# Verify user credentials by checking the hashed password
def check_user_credentials(email, password):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('SELECT password FROM users WHERE email = ?', (email,))
            result = c.fetchone()
            if result and check_password_hash(result[0], password):
                return True
            return False
    except Exception as e:
        print(f"Error checking credentials: {e}")
        return False

# Find a user by their email address
def find_user_by_email(email):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('SELECT id, email, password FROM users WHERE email = ?', (email,))
            return c.fetchone()
    except Exception as e:
        print(f"Error finding user: {e}")
        return None

# Update user's password
def update_user_password(email, new_password):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            hashed_password = generate_password_hash(new_password)
            c.execute('UPDATE users SET password = ? WHERE email = ?', (hashed_password, email))
            conn.commit()
            return c.rowcount > 0  # Return True if the update was successful
    except Exception as e:
        print(f"Error updating password: {e}")
        return False

# Save feedback to the database
def save_feedback(user_id, feedback_text):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO feedback (user_id, feedback_text) VALUES (?, ?)', (user_id, feedback_text))
            conn.commit()
    except Exception as e:
        print(f"Error saving feedback: {e}")

# Get all feedback from the database
def get_all_feedback():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''SELECT feedback.feedback_text, feedback.timestamp, users.email
                         FROM feedback
                         INNER JOIN users ON feedback.user_id = users.id
                         ORDER BY feedback.timestamp DESC''')
            return c.fetchall()
    except Exception as e:
        print(f"Error retrieving feedback: {e}")
        return []

# Function to save the filter applied with video_id and timestamp
def save_detected_filter(user_email, detected_filter, video_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()

            # Get the current timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Insert the detected filter into the filters table
            c.execute('''
                INSERT INTO filters (user_email, filter, video_id, timestamp) 
                VALUES (?, ?, ?, ?)
            ''', (user_email, detected_filter, video_id, timestamp))

            conn.commit()
            print(f"Filter {detected_filter} saved for user_email {user_email} for video {video_id} at {timestamp}")
            return True
    except Exception as e:
        print(f"Error saving filter: {e}")
        return False
