import cv2
from deepface import DeepFace
import pyttsx3
import speech_recognition as sr
import time
import pandas as pd
import ast
from gtts import gTTS
from playsound import playsound
import tempfile
import os
import uuid
from sklearn.cluster import KMeans
import numpy as np
import google.generativeai as genai
from langdetect import detect
from googletrans import Translator
import sys

def get_resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return filename

csv_path = get_resource_path("output.csv")
response_df = pd.read_csv(csv_path)

# === SET YOUR GEMINI API KEY HERE ===
GEMINI_API_KEY = "AIzaSyCbMtuuJ3djXaOzoyXeum6hEA0LIdpXv0k"  
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# === Text-to-Speech Setup ===
engine = pyttsx3.init()
recognizer = sr.Recognizer()
translator = Translator()

def speak(text, lang='en'):
    print(f"🤖 Speaking ({lang}): {text}")
    if lang != 'en':
        try:
            filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
            gTTS(text=text, lang=lang).save(filename)
            playsound(filename)
            os.remove(filename)
        except:
            engine.say(text)
            engine.runAndWait()
    else:
        engine.say(text)
        engine.runAndWait()

def listen(timeout=60, phrase_time_limit=45):
    with sr.Microphone() as source:
        print(f"🎤 Listening (timeout={timeout}, phrase_time_limit={phrase_time_limit})...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            query = recognizer.recognize_google(audio)
            print("You said:", query)
            return query.lower()
        except sr.WaitTimeoutError:
            print("⏰ Listening timed out.")
            return ""
        except sr.UnknownValueError:
            print("🤷 Couldn't understand.")
            return ""
        except Exception as e:
            print(f"⚠️ Error: {e}")
            return ""

# Prepare vector clustering
vector_data = np.array([ast.literal_eval(v) if isinstance(v, str) else v for v in response_df['vector']])
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(vector_data)
response_df['cluster'] = kmeans.labels_

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found!")
    exit()

start_time = time.time()
last_detection_time = 0
cooldown = 5 * 60
initial_wait_done = False

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    current_time = time.time()
    cv2.imshow("🧠 Emotion Bot (Press Q to Quit)", frame)

    should_detect = (not initial_wait_done and current_time - start_time >= 5) or \
                    (initial_wait_done and current_time - last_detection_time >= cooldown)

    if should_detect:
        initial_wait_done = True
        last_detection_time = current_time
        try:
            speak("Hello😄, I can see you..")
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='retinaface')
            if isinstance(results, dict):
                results = [results]

            for result in results:
                if 'dominant_emotion' not in result or 'region' not in result:
                    continue

                region = result['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                emotion = result['dominant_emotion'].lower()

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print(f"You are {emotion}")

                # Auto Language Detection
                speak("Please say something so I can detect your language.", lang='en')
                lang_input = listen()
                lang = detect(lang_input) if lang_input else 'en'

                try:
                    lang_name = translator.translate("language", dest=lang).src
                except:
                    lang_name = lang

                speak(f"Great! I will talk to you in {lang_name}.", lang=lang)

                # Optional intro from CSV
                embedding = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)[0]['embedding']
                cluster_id = kmeans.predict([embedding])[0]
                matching_rows = response_df[response_df['cluster'] == cluster_id]
                matching_rows = matching_rows.dropna(subset=[f'message_{i}' for i in range(20)])
                intro_msg = None

                if not matching_rows.empty:
                    row = matching_rows.sample(1).iloc[0]
                    messages = [row[f'message_{i}'] for i in range(20) if pd.notna(row.get(f'message_{i}'))]
                    if messages:
                        intro_msg = np.random.choice(messages)

                if intro_msg:
                    speak(intro_msg, lang=lang)

                # GEMINI Conversation
                no_response_count = 0
                conversation = [f"The user seems to be feeling {emotion}."]
                if intro_msg:
                    conversation.append(intro_msg)

                first_response = True

                while True:
                    if no_response_count >= 3:
                        speak("Seems you're not in the mood to talk. I'll check again later.", lang=lang)
                        break

                    speak("I'm listening...", lang=lang)
                    user_input = listen()

                    if not user_input:
                        no_response_count += 1
                        continue

                    conversation.append(f"User: {user_input}")

                    if first_response:
                        full_prompt = f"Reply in {lang} language. The user is feeling {emotion}. Asking about {emotion}.\n\n" + "\n".join(conversation)
                        first_response = False
                    else:
                        full_prompt = "\n".join(conversation) + "\nPlease reply in 1 to 10 sentences only. Reply in " + lang + "."

                    try:
                        gemini_response = gemini_model.generate_content(full_prompt)
                        reply = gemini_response.text.strip()

                        # Translate if needed
                        if lang != 'en':
                            try:
                                translated = translator.translate(reply, dest=lang)
                                reply = translated.text
                            except:
                                pass

                        speak(reply, lang=lang)
                        conversation.append(f"Bot: {reply}")
                        no_response_count = 0
                    except Exception as e:
                        print("❌ Gemini Error:", e)
                        speak("Sorry, Gemini could not reply.", lang=lang)
                        break

        except Exception as e:
            print("❌ Detection Error:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()


