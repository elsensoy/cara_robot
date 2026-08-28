import json
import os
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# --- 0. CONFIGURATION ---
# Define where these memories will be saved
MEMORY_DIR = "cara_memory"
SESSIONS_DIR = os.path.join(MEMORY_DIR, "sessions")

# --- 1. THE SAVE FUNCTION (Implemented directly here) ---
def save_session_entry(summary, emotion, topics, user_quotes, cara_support, tags):
    """
    Saves the summarized session data to a JSON file.
    """
    # 1. Ensure the folder exists
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    # 2. Create the filename (e.g., session_20251219_103000.json)
    timestamp = datetime.now()
    filename = f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(SESSIONS_DIR, filename)

    # 3. Construct the data packet
    entry_data = {
        "timestamp": timestamp.isoformat(),
        "summary": summary,
        "emotion": emotion,
        "topics": topics,
        "user_quotes": user_quotes,
        "cara_support": cara_support,
        "tags": tags
    }

    # 4. Write to disk
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entry_data, f, indent=2, ensure_ascii=False)
        return file_path
    except Exception as e:
        print(f"Failed to write session file: {e}")
        return None

# --- 2. Helper for ISO timestamp ---
def current_iso_timestamp():
    return datetime.now().replace(microsecond=0).isoformat()

# --- 3. LLM client wrapper ---
def call_llm_for_summary(system_prompt: str, instruction_text: str, conversation_log):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found.")
        return "{}"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Updated to a standard model name
        system_instruction=system_prompt
    )

    try:
        response = model.generate_content(
            instruction_text,
            generation_config={"response_mime_type": "application/json"}
        )
        return response.text
    except Exception as e:
        print(f"Summarization Logic Error: {e}")
        return "{}"

# --- 4. Main function: summarize + save ---
def summarize_conversation_for_memory(conversation_log):
    """
    Takes the conversation log, summarizes it via Gemini, and saves it.
    Returns: The file path of the saved memory.
    """
    
    # A. Define the Prompt
    system_prompt = (
        "You are a memory-summarizing assistant for an AI teddy bear companion named Cara.\n"
        "Your task:\n"
        "- Read a full conversation between the user and Cara.\n"
        "- Extract the emotionally and personally important parts.\n"
        "- Output a single JSON object describing this conversation.\n"
    )

    instruction_text = (
        "I will now give you a JSON array called \"conversation_log\".\n"
        "Please output a single JSON object with EXACTLY the following keys:\n"
        "{\n"
        "  \"summary\": \"<1-3 sentences>\",\n"
        "  \"emotion\": \"<short description>\",\n"
        "  \"topics\": [\"<topic1>\", \"<topic2>\"],\n"
        "  \"user_quotes\": [\"<quote1>\", \"<quote2>\"],\n"
        "  \"cara_support\": \"<1-3 sentences>\",\n"
        "  \"tags\": [\"<tag1>\", \"<tag2>\"]\n"
        "}\n\n"
        "Here is the conversation_log:\n"
        + json.dumps({"conversation_log": conversation_log}, ensure_ascii=False, indent=2)
    )

    # B. Get Summary from AI
    print("Summarizing session...")
    raw_reply = call_llm_for_summary(system_prompt, instruction_text, conversation_log)

    # C. Parse JSON
    try:
        data = json.loads(raw_reply)
    except json.JSONDecodeError:
        print(f"JSON Parse Error. Raw reply was: {raw_reply}")
        return None

    # D. Save using the function defined above
    path = save_session_entry(
        summary=data.get("summary", "No summary generated."),
        emotion=data.get("emotion", "Unknown"),
        topics=data.get("topics", []),
        user_quotes=data.get("user_quotes", []),
        cara_support=data.get("cara_support", ""),
        tags=data.get("tags", []),
    )

    return path
