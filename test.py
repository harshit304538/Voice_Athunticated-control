import os
import pandas as pd
import numpy as np
import librosa
import requests
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write
from scipy.spatial.distance import euclidean
import tkinter as tk
from tkinter import messagebox, simpledialog

# Constants
csv_path = 'voice_data.csv'
duration = 2
sample_rate = 44100

# ESP32/ESP8266 base URL (update this IP to your actual device IP)
BASE_URL = " IP ADDRESS OF YOUR DEVICE "

COMMANDS = {
    # LED 1
    "turn on led 1": f"{BASE_URL}/LED1=HIGH",
    "turn off led 1": f"{BASE_URL}/LED1=LOW",
    "turn on led one": f"{BASE_URL}/LED1=HIGH",
    "turn off led one": f"{BASE_URL}/LED1=LOW",

    # LED 2
    "turn on led 2": f"{BASE_URL}/LED2=HIGH",
    "turn off led 2": f"{BASE_URL}/LED2=LOW",
    "turn on led two": f"{BASE_URL}/LED2=HIGH",
    "turn off led two": f"{BASE_URL}/LED2=LOW",

    # LED 3
    "turn on led 3": f"{BASE_URL}/LED3=HIGH",
    "turn off led 3": f"{BASE_URL}/LED3=LOW",
    "turn on led three": f"{BASE_URL}/LED3=HIGH",
    "turn off led three": f"{BASE_URL}/LED3=LOW",

    # LED 4
    "turn on led 4": f"{BASE_URL}/LED4=HIGH",
    "turn off led 4": f"{BASE_URL}/LED4=LOW",
    "turn on led four": f"{BASE_URL}/LED4=HIGH",
    "turn off led four": f"{BASE_URL}/LED4=LOW",

    # All Off
    "turn off all": f"{BASE_URL}/LED=OFF",
    "turn off": f"{BASE_URL}/LED=OFF"
}

# Load or initialize DataFrame
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df["mfccs"] = df["mfccs"].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
else:
    df = pd.DataFrame(columns=["pitch_hz", "loudness_db", "mfccs"])



def speech_to_text():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use microphone as source
    with sr.Microphone() as source:

        audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)  # Listen for audio

        try:
            # Recognize speech using Google's free Web API
            text = recognizer.recognize_google(audio)
            return f"Voice Command:  {text}\n"
        except sr.UnknownValueError:
            return "Sorry, could not understand the audio."
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition service."


# Helper Functions

#DELET THE AUDIO FILE
def deletefile(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

#EXTRACT THE FEATURE OF THE AUDIO FILE
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)

    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_hz = np.nanmean(f0)
    rms = librosa.feature.rms(y=y)
    loudness_db = np.mean(librosa.amplitude_to_db(rms))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    avg_mfccs = np.mean(mfccs, axis=1)
    return {"pitch_hz": pitch_hz, "loudness_db": loudness_db, "mfccs": avg_mfccs}

# LOAD DATA OF THE AUDIO TO CSV FILE
def add_dataframe(file_path, user_name):
    global df
    features = analyze_audio(file_path)
    features["mfccs"] = np.array2string(features["mfccs"], separator=' ', precision=2)
    df.loc[user_name] = features
    df.to_csv(csv_path)

#COMPARING THE DATA OF AUDIO FILE 
def compare_audio_features(feat1, feat2):
    
    # check if the mfccs value are store as a string and 
    # convert them into numpy array for better calculations
    if isinstance(feat1["mfccs"], str):             
        feat1["mfccs"] = np.fromstring(feat1["mfccs"].strip("[]"), sep=' ')
    if isinstance(feat2["mfccs"], str):
        feat2["mfccs"] = np.fromstring(feat2["mfccs"].strip("[]"), sep=' ')


    #calculate the difference between pitch , loudness and mfcc of two audio files
    pitch_diff = abs(feat1["pitch_hz"] - feat2["pitch_hz"])
    loudness_diff = abs(feat1["loudness_db"] - feat2["loudness_db"])
    mfcc_distance = euclidean(feat1["mfccs"], feat2["mfccs"])
    return pitch_diff, loudness_diff, mfcc_distance

#calculate how much similar two files are
def similarity_score(pitch_diff, loudness_diff, mfcc_dist):

    # Normalize the differences to a scale of 0 to 1
    pitch_score = max(0, 1 - (pitch_diff / 30))
    loudness_score = max(0, 1 - (loudness_diff / 10))
    mfcc_score = max(0, 1 - (mfcc_dist / 100))
    # Combine the scores with weights
    # You can adjust the weights based on your preference
    return (pitch_score * 0.3 + loudness_score * 0.2 + mfcc_score * 0.5) * 100

#comparison function
#compare the audio features of the user with the new audio file
def comparison(features):
    similarity = []
    for index, row in df.iterrows():
        pitch_diff, loudness_diff, mfcc_dist = compare_audio_features(row, features.iloc[0])
        score = similarity_score(pitch_diff, loudness_diff, mfcc_dist)
        similarity.append((index, score))
     # Show the most similar rows (you can change the number of results shown)
    print("Most similar audio features:")
    for idx, score in similarity:
        print(f"Index: {idx}, Similarity: {score:.2f}%")
        if(score > 70.00):        #check the similarity score
            return "yes",idx,f" Similarity: {score:.2f}%"
            break
        
    return "no","No One", f" Similarity: {score:.2f}%"



# GUI Functions

# Add a new user
# This function will record the user's voice and save it to a file
def add_user():
    name = simpledialog.askstring("Add User", "Enter user's name:")
    if not name:
        return
    for i in range(3):
        messagebox.showinfo("Recording", "Recording will start. Please speak now...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
        sd.wait()
        file_path = f"{name}{i}_temp.wav"
        write(file_path, sample_rate, audio)
        add_dataframe(file_path, name+str(i))
        deletefile(file_path)

    messagebox.showinfo("Success", f"User '{name}' added.")

# Delete a user
# This function will delete the user's data from the CSV file
def delete_user():
    if df.empty:
        messagebox.showinfo("Info", "No users to delete.")
        return
    users = "\n".join(df.index[::3])
    user_list = "\n".join(users[:-1] for users in df.index[::4])
    name = simpledialog.askstring("Delete User", f"Enter user's name to delete:\n\nAvailable:\n{user_list}")
    if not name:
        return
     
    for i in df.index:
        if (i[:-1]==name):
            df.drop(index=i, inplace=True)
            df.to_csv(csv_path)
    if name in user_list:
        messagebox.showinfo("Deleted", f"User '{name}' deleted.")
    else:    
        messagebox.showwarning("Error", "User not found.")

# Compare voice
# This function will record a voice command and compare it with stored users
def compare_voice():
    messagebox.showinfo("Recording", "Recording for comparison. Please speak now...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
    #recording is a asyncronized process therefore we can convert the audio to text parallelly with recording
    text = speech_to_text()
    command_text=text.lower()
    temp_file = "temp_compare.wav"
    write(temp_file, sample_rate, audio)
    features = analyze_audio(temp_file)
    features["mfccs"] = np.array2string(features["mfccs"], separator=' ', precision=2)
    features_df = pd.DataFrame([features])
    result, user, score = comparison(features_df)
    #after comparing the voice of the user with the data in the csv file delete the temp file
    deletefile(temp_file)
    # Extract the name from the user variable
    # The user variable contains the name and a number (e.g., "John0")
    # We want to extract just the name part
    # For example, if user is "John0", we want to get "John"
    # This is done by slicing the string to remove the last character
    name = user[:-1]
    
    # Check if the result is "yes" and if the command is recognized
    # If the result is "yes", we will check if the command is recognized
    if result == "yes":
        for key in COMMANDS:
            # Check if the command text contains any of the keys in COMMANDS
            # If it does, we will send the command to the the connecte server
            if key in command_text:
               #✅ Recognized command: {key}
                try:
                    # Send the command to the server
                    # The server will execute the command and return a response
                    response = requests.get(COMMANDS[key], timeout=5)
                    # Check if the response is successful
                    if response.status_code == 200:
                        # Show a message box with the result
                        # The message box will show the name of the user, the command, and the score of similRity
                        # The message box will also show the command that was sent to the server
                        messagebox.showinfo("Match Found", f"  Voice matches with: {name}\n "
                                            f"✅ Recognized command: {key} \n with score {score}\n\n {command_text} \n"
                                            f"🌐 Command sent successfully: {COMMANDS[key]}")
                        # If the command is recognized and sent successfully, we can break the loop
                        break   
                        #🌐 Command sent successfully: {COMMANDS[key]}
                    else:
                        # If the response is not successful, show an error message
                        messagebox.showinfo( "Error",f"⚠️ Failed to send command. \nStatus code: {response.status_code}"
                                            f"\n Command:{command_text}")
                except requests.exceptions.Timeout:
                    # If the request times out, show an error message
                    # The error message will show the command that was sent to the server
                    messagebox.showerror("Timeout", f"❌ Request timed out while sending:\n {command_text}")                  
                except requests.exceptions.RequestException as e:  
                    # If there is any other request exception, show an error message
                    # The error message will show the error that occurred
                    messagebox.showerror("Timeout", f"❌ Request timed out while sending:\n {command_text}")
                    messagebox.showinfo("Error",f"🚫 Error sending request: {e}\nCommand:{command_text}")
                return
        messagebox.showinfo("Command pannel.", f"Command not recoganised ❓\nVoice matches with: {name}\n  command: {command_text}")

        #messagebox.showinfo("Match Found", f"Voice matches with: {name} \nwith score {score}\n\n {text}")
    else:
        # If the result is "no", show a message box indicating no match
        messagebox.showinfo("No Match", "Voice does not match any stored user.")




# Build GUI
app = tk.Tk()
app.title("Voice Recognition App")
app.geometry("300x250")

tk.Label(app, text="Voice Recognition System", font=("Helvetica", 14, "bold")).pack(pady=10)
tk.Button(app, text="Add New User", width=25, command=add_user).pack(pady=5)
tk.Button(app, text="Delete User", width=25, command=delete_user).pack(pady=5)
tk.Button(app, text="Give Command", width=25, command=compare_voice).pack(pady=10)
tk.Button(app, text="Exit", width=25, command=app.quit).pack(pady=20)

app.mainloop()
