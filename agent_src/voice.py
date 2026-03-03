def record_and_transcribe() -> str:
    """
    Record audio and transcribe it.
    Stops when "NEXT QUESTION" is detected.
    
    Returns:
        Transcribed text
    
    Requires: pip install SpeechRecognition pyaudio
    """
    import speech_recognition as sr
    
    recognizer = sr.Recognizer()
    
    print("🎤 Recording... (say 'NEXT QUESTION' to finish)")
    full_transcript = []
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        while True:
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)
                text = recognizer.recognize_google(audio)
                print(f"   You said: {text}")
                
                # Check for stop phrase
                if "next question" in text.lower():
                    print("   ✓ Moving to next question!")
                    # Remove "next question" from the transcript
                    text = text.lower().replace("next question", "").strip()
                    if text:
                        full_transcript.append(text)
                    break
                
                full_transcript.append(text)
            except sr.UnknownValueError:
                print("   Could not understand, continue speaking or say 'NEXT QUESTION'...")
                continue
            except sr.RequestError as e:
                return f"Error: {e}"
    
    return " ".join(full_transcript) if full_transcript else "No speech detected"


def wait_for_wake_word():
    """Wait for 'Dominica' wake word before starting"""
    import speech_recognition as sr
    
    recognizer = sr.Recognizer()
    
    print("\n👂 Listening for wake word 'Dominica'...")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=3)
                text = recognizer.recognize_google(audio).lower()
                print(f"   Heard: {text}")
                if "dominica" in text.lower():
                    print("\n✨ Wake word detected! Let's build your profile!\n")
                    return True
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"   Error: {e}")
                continue
