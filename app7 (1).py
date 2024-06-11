import streamlit as st
from streamlit_calendar import calendar
import calendar as builtin_calendar
from datetime import datetime, timedelta
import schedule
import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from twilio.rest import Client
import os
import gdown
from transformers import pipeline

account_sid = "AC7a85ece0348d02be0f38040d9dfc8926"
auth_token = "573b91b2cbb0162e41ce95808b6bfd72"
client = Client(account_sid, auth_token)

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

model = YOLO(model_path)

folder_id = "1w2-zNG0mUt6SufkO1YvoC-KsqQX7DNaR"

local_directory = r"C:\Users\asuss\OneDrive\Pictures"
os.makedirs(local_directory, exist_ok=True)

gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", output=local_directory)

scheduled_runs = []
scheduled_tasks = {}  
scheduled_activities = {}  

def run_script():
    global scheduled_runs  
    face_detected_images = []
    face_detected = False

    
    for filename in os.listdir(local_directory):
        
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        
            image_path = os.path.join(local_directory, filename)

            image = Image.open(image_path)


            output = model(image)

            # Parse results
            results = Detections.from_ultralytics(output[0])

            # Check if any faces were detected
            if len(results) > 0:
                face_detected = True
                face_detected_images.append(image)

    # Send a message using Twilio if at least one face was detected
    if face_detected:
        message = client.messages.create(
            to="+919698095155",
            from_="+15169904604",
            body="A face was detected in one or more images."
        )
        st.write(f"Message sent: {message.sid}")

        # Display the images where faces were detected
        st.image(face_detected_images, caption="Face detected", use_column_width=True)

def main():
    global scheduled_runs, scheduled_tasks, scheduled_activities  # Declare global variables

    # Load custom CSS styles
    with open(r"C:\Users\asuss\Downloads\styles") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Render the title with inline HTML styles
    st.markdown(f"<h1 style='font-size: 1.875rem; font-weight: bold; color: #3b82f6;'>Web Monitor</h1>", unsafe_allow_html=True)

    # Display the calendar
    selected_date = st.date_input("Select Month and Year", value=datetime.now(), key="calendar_date")
    current_month = selected_date.month
    current_year = selected_date.year

    # Create a realistic image representation of the calendar
    calendar_output = st.empty()
    calendar_html = calendar(current_year, current_month, key="calendar")
    calendar_output.write(calendar_html, unsafe_allow_html=True)

    # Input for desired run date and time
    run_date = st.date_input("Enter the desired run date")
    run_time = st.time_input("Enter the desired run time", value=None)
    task_description = st.text_input("Enter task description (optional)")

    schedule_activity = st.button("Schedule Activity")
    if schedule_activity and run_date and run_time:
        # Schedule the script to run at the desired date and time
        schedule_time = datetime.combine(run_date, run_time)
        scheduled_runs.append(schedule_time)
        schedule.every().day.at(schedule_time.strftime("%H:%M")).do(run_script)

        # Store the scheduled task
        scheduled_tasks[schedule_time] = task_description

        # Display the upcoming scheduled runs as a list within the calendar section
        scheduled_runs_list = calendar_output.markdown("<div class='scheduled-runs'><h3>Upcoming Scheduled Runs:</h3><ul></ul></div>", unsafe_allow_html=True)
        for scheduled_run in sorted(scheduled_runs):
            task_desc = scheduled_tasks.get(scheduled_run, "")
            scheduled_runs_list.write(f"<li>{scheduled_run.strftime('%Y-%m-%d %H:%M')} - {task_desc}</li>", unsafe_allow_html=True)

    # Custom button for scheduling additional activities
    activity_name = st.text_input("Enter activity name")
    activity_description = st.text_area("Enter activity description")
    schedule_additional_activity = st.button("Schedule Additional Activity")
    if schedule_additional_activity and activity_name and activity_description:
        scheduled_activities[activity_name] = activity_description
        st.success(f"Activity '{activity_name}' scheduled successfully!")

    # Text box for query
    query = st.text_area("Enter your query here:")

    # Load KissanAI model
    kissan_ai = pipeline("text-generation", model="PhurieTech/kissan")

    # Generate text using KissanAI
    if query:
        output_text = kissan_ai(query, max_length=500, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]['generated_text']
        st.write(f"Generated text: {output_text}")

    # Custom button for checking scheduled activities and running the script
    check_activity = st.button("Check Activity")
    if check_activity:
        run_script()

        # Check and display scheduled activities
        if scheduled_activities:
            st.write("Scheduled Activities:")
            for activity_name, activity_description in scheduled_activities.items():
                st.write(f"- **{activity_name}**: {activity_description}")
        else:
            st.write("No scheduled activities found.")

    # Custom button for stopping all scheduled runs
    stop_all = st.button("Stop All Scheduled Runs")
    if stop_all:
        schedule.clear()
        scheduled_runs = []
        scheduled_tasks = {}
        st.write("All scheduled runs have been stopped.")

    # Custom button for pausing all scheduled runs
    pause_all = st.button("Pause All Scheduled Runs")
    if pause_all:
        schedule.pause()
        st.write("All scheduled runs have been paused.")

    # Run the scheduled tasks
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
