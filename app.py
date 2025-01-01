from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from database import init_db, add_user, check_user_credentials, update_user_password, find_user_by_email, save_feedback, get_all_feedback,save_detected_filter, DB_PATH
import cv2
from ultralytics import YOLO as YOLOv8
import os
import threading
from datetime import datetime
import sqlite3
import logging


# Flask App Initialization
app = Flask(__name__)
app.secret_key = 'your_secret_key'

init_db()

# Initialize YOLOv8 model
model = YOLOv8(r'D:\Final Year Project\videos_analyzer\train7\train7\weights\best.pt')

# Configure upload and detected video folders
UPLOAD_FOLDER = 'uploads'
DETECTED_VIDEO_FOLDER = r'D:\Final Year Project\videos_analyzer\detected_videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_VIDEO_FOLDER'] = DETECTED_VIDEO_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_VIDEO_FOLDER'], exist_ok=True)

# Dummy in-memory user database
users = {}

# Progress data dictionary
progress_data = {}

# Modify the process_video function to automatically detect and save filters
def process_video(video_path, output_path, video_id, user_email):
    print(f"Processing video: {video_path}")
    print(f"Saving processed video to: {output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        progress_data[video_id] = -1  # Indicate failure
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Error: Could not initialize VideoWriter.")
        progress_data[video_id] = -1  # Indicate failure
        return

    processed_frames = 0
    detected_filters = set()  # Set to store the detected filters

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLOv8 inference
        try:
            results = model(frame)
            keep_frame = True  # Assume the frame is valid unless inappropriate content is detected

            # Check each detected object in the frame
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for class_id in class_ids:
                    detected_name = model.names[int(class_id)]
                    print(f"Detected: {detected_name}")

                    # Apply filter based on detected name
                    if detected_name in ['nude', 'profanity', 'kiss', 'bad_gesture', 'violence']:
                        keep_frame = False  # Skip frames that contain inappropriate content
                        detected_filters.add(detected_name)  # Add the detected filter to the set
                        break  # Stop further processing for this frame

                if not keep_frame:
                    break  # Skip further processing if frame is marked for removal

            if keep_frame:
                out.write(frame)  # Write valid frames only
        except Exception as e:
            print(f"Error during YOLO detection: {e}")

        processed_frames += 1
        progress_data[video_id] = int((processed_frames / total_frames) * 100)

    # Save all the detected filters for the user
    for detected_filter in detected_filters:
        save_detected_filter(user_email, detected_filter, video_id)

    cap.release()
    out.release()
    progress_data[video_id] = 100  # Mark as complete

    print("Processing complete.")

# Routes for Signup, Login, Logout
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if add_user(email, password):
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        else:
            flash('User already exists.', 'danger')
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if check_user_credentials(email, password):
            session['user'] = email
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# Home Route for Video Upload and Processing
# In the home route
@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'user' not in session:
        return redirect(url_for('signup'))

    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['video']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            # Generate a unique ID for the video processing
            video_id = datetime.now().strftime('%Y%m%d%H%M%S')
            unique_filename = f"{video_id}_{file.filename}"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(video_path)

            # Prepare output path
            output_filename = f"{video_id}_processed.mp4"
            output_path = os.path.join(app.config['DETECTED_VIDEO_FOLDER'], output_filename)

            # Ensure user_email is passed correctly
            user_email = session.get('user')

            # Start background processing and pass the user_email explicitly
            threading.Thread(target=process_video, args=(video_path, output_path, video_id, user_email)).start()

            return jsonify({'message': 'Video is being processed', 'video_id': video_id})

    return render_template('index.html')


# Route for Checking Processing Progress
@app.route('/progress/<video_id>')
def get_progress(video_id):
    progress = progress_data.get(video_id, 0)  # Default to 0 if not found
    return jsonify({'progress': progress})

# Route for Downloading Processed Video
@app.route('/download_video/<video_id>', methods=['GET'])
def download_video(video_id):
    try:
        video_filename = f"{video_id}_processed.mp4"
        video_path = os.path.join(app.config['DETECTED_VIDEO_FOLDER'], video_filename)

        print(f"Looking for file: {video_path}")

        if not os.path.exists(video_path):
            return jsonify({"message": "No processed video available"}), 404

        return send_file(video_path, as_attachment=True)
    except Exception as e:
        print(f"Error during video download: {e}")
        return jsonify({"message": "An error occurred while processing your request."}), 500

# Add a new route for the subscription page
@app.route('/subscription', methods=['GET'])
def subscription():
    if 'user' not in session:
        flash('You need to log in to view the subscription page.', 'warning')
        return redirect(url_for('login'))

    return render_template('subscription.html')


# Add a new route for the View Rendering page
@app.route('/view-rendering', methods=['GET'])
def view_rendering():
    # Check if the user is logged in
    if 'user' not in session:
        flash('You need to log in to view the subscription page.', 'warning')
        return redirect(url_for('view_rendering'))

    # Render the view-rendering page
    return render_template('view-rendering.html')

# Add a new route for the View Rendering page
@app.route('/license-agreement', methods=['GET'])
def license_agreement():
    # Check if the user is logged in
    # Render the view-rendering page
    return render_template('license-agreement.html')

# Add a new route for the pricing 
@app.route('/pricing', methods=['GET'])
def pricing():
    # Check if the user is logged in
    # Render the view-rendering page
    return render_template('pricing.html')



# Add a new route for the View Rendering page
@app.route('/contact', methods=['GET'])
def contact():
    # Check if the user is logged in
    # Render the view-rendering page
    return render_template('contact.html')

# Forgot Password Route
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = find_user_by_email(email)

        if user:
            return redirect(url_for('reset_password', email=email))
        else:
            flash('Email not found', 'danger')

    return render_template('forgot-password.html')


# Reset Password Route
@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    email = request.args.get('email')
    if not email:
        flash('Invalid request', 'danger')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        if new_password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('reset_password', email=email))

        if update_user_password(email, new_password):
            flash('Password has been successfully updated.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Failed to update password. Please try again.', 'danger')
            return redirect(url_for('reset_password', email=email))

    return render_template('reset-password.html', email=email)

# Feedback Route
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user' not in session:
        flash('You need to log in to submit feedback.', 'warning')
        return redirect(url_for('login'))

    user = find_user_by_email(session['user'])
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        feedback_text = request.form.get('feedback', '').strip()
        if not feedback_text:
            flash('Feedback cannot be empty.', 'danger')
            return redirect(url_for('feedback'))

        save_feedback(user[0], feedback_text)  # `user[0]` is the user's ID
        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('feedback'))

    feedbacks = get_all_feedback()
    return render_template('feedback.html', feedbacks=feedbacks)

#  When a filter is applied after video processing
@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    try:
        data = request.get_json()
        user_email = session.get('user')
        if not user_email:
            return jsonify({'error': 'User not logged in'}), 401
        
        filter_name = data.get('filter')
        video_id = data.get('video_id')  # Assume video_id is passed along with the filter

        # Save the filter with the specific video_id
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO filters (user_email, filter, video_id, timestamp) 
                VALUES (?, ?, ?, ?)
            ''', (user_email, filter_name, video_id, datetime.datetime.now()))
            conn.commit()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f"Error applying filter: {e}"}), 500

#  History of All Filters and user
@app.route('/get_filters', methods=['GET'])
def get_filters():
    user_email = session.get('user')  # Ensure session has the logged-in user
    if not user_email:
        return jsonify({'error': 'User not logged in'}), 401  # Return error if not logged in

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''SELECT filter, timestamp, user_email 
                         FROM filters 
                         WHERE user_email = ? 
                         ORDER BY timestamp DESC''', (user_email,))

            filters = c.fetchall()

            if filters:
                # Format the response to include user_email for each filter
                filters_data = [{"filter": row[0], "timestamp": row[1], "user_email": row[2]} for row in filters]
                return jsonify({'filters': filters_data})
            else:
                return jsonify({'message': 'No filters found for this user'}), 404
    except Exception as e:
        return jsonify({'error': f"Error fetching filters: {e}"}), 500

  
#  get applied filter by user 
@app.route('/get_applied_filter/<video_id>', methods=['GET'])
def get_applied_filter(video_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()

            # Query to get the filter applied for a specific video_id
            c.execute('''SELECT filter, timestamp 
                         FROM filters 
                         WHERE video_id = ?''', 
                         (video_id,))

            result = c.fetchall()
            
            print(f"Filters for video_id {video_id}: {result}")  # Debugging line

            if result:
                filters = [{"filter": row[0], "timestamp": row[1], "user_email": row[2]} for row in result]
                return jsonify({"filters": filters})
            else:
                return jsonify({"message": "No filters applied for this video."})
    except Exception as e:
        print(f"Error fetching applied filter: {e}")
        return jsonify({"error": "Internal server error"}), 500



# Default Route
@app.route('/')
def index():
    return redirect(url_for('signup'))


# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
