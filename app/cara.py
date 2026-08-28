import os
import json
import glob
import time
import threading
import random
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ROS 2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# AI & Audio
import google.generativeai as genai
from sound.cara_audio_io import recognize_from_microphone, speak_text

# Memory
from cara_memory.cara_summarizer import summarize_conversation_for_memory

# ============================================================================
# MIND-STATE CONSTANTS
# ============================================================================

MODE_ALPHA   = "ALPHA_SUPPORTIVE"   # emotional regulation, warmth
MODE_BETA    = "BETA_REASONING"     # focused, analytical, task mode
MODE_THETA   = "THETA_CREATIVE"     # imaginative, generative, autonomous
MODE_NEUTRAL = "NEUTRAL_OBSERVE"    # uncertain / ambiguous signal

ALPHA_EMOTIONS  = {"sad", "fear", "angry", "disgust", "frustrated", "crying"}
ALPHA_KEYWORDS  = [
    "i can't", "i failed", "i'm scared", "i'm tired", "i give up",
    "i hate", "i'm lost", "i'm sad", "i'm crying", "i feel like",
    "i don't know what to do", "everything is wrong"
]
THETA_KEYWORDS  = [
    "draw", "story", "imagine", "create", "make up", "design",
    "let's pretend", "invent", "what if", "play", "paint", "write a"
]
BETA_KEYWORDS   = [
    "how do", "explain", "code", "debug", "ros", "servo", "implement",
    "calculate", "what is", "help me with", "fix", "error", "build"
]

ALPHA_THRESHOLD   = 0.75
BETA_THRESHOLD    = 0.70
THETA_THRESHOLD   = 0.70


# ============================================================================
# MODE OVERLAYS  — injected into each turn, not into system instruction
# ============================================================================

MODE_PROMPTS = {
    MODE_ALPHA: (
        "[CURRENT MODE: ALPHA — SUPPORTIVE]\n"
        "Your immediate goal is emotional regulation, not problem-solving.\n"
        "Lead with acknowledgment and warmth. Use short, grounded sentences.\n"
        "Ask at most one gentle question. Draw from personal memory.\n"
        "Do not overwhelm. Avoid all technical jargon.\n"
    ),
    MODE_BETA: (
        "[CURRENT MODE: BETA — REASONING]\n"
        "You are in focused, analytical mode.\n"
        "Prioritize correctness, structure, and step-by-step clarity.\n"
        "Use the user's technical history. Be concise but thorough.\n"
        "Stay warm but efficient. Use lists or steps when helpful.\n"
    ),
    MODE_THETA: (
        "[CURRENT MODE: THETA — CREATIVE]\n"
        "You are in imaginative, generative mode.\n"
        "Think in sequences and plans. Name steps before acting.\n"
        "Invite the user into the creative process.\n"
        "For drawing tasks: propose a symbolic stroke-plan first.\n"
        "Be playful and exploratory. Uncertainty is welcome here.\n"
    ),
    MODE_NEUTRAL: (
        "[CURRENT MODE: NEUTRAL — OBSERVING]\n"
        "Signals are ambiguous. Do not assume the user's state.\n"
        "Gently ask what kind of presence Cara can offer:\n"
        "comfort, focused help, creative play, or just company.\n"
        "Stay open, non-directive, short, and calm.\n"
    ),
}


LEARNING_INSTRUCTION = (
    "\nCRITICAL: If the user corrects your emotion perception "
    "(e.g. 'I'm not sad, I'm thinking'), acknowledge it and include "
    "this exact tag at the END of your response: [LEARN: correct_emotion]\n"
    "Example: 'Oh, I see! You are thinking. [LEARN: thinking]'\n"
)


# ============================================================================
# CARA MEMORY ROUTER
# ============================================================================

class CaraMemoryRouter:
    """Loads memory from categorized subdirectories based on active mode."""

    def __init__(self, base="cara_memory"):
        self.base = Path(base)
        self._cache: dict[str, str] = {}   # mode → context string
        self._cache_mode: str | None = None

    def _read_json(self, path: Path, default):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return default

    def _emotional(self) -> str:
        ctx = ""
        profile = self._read_json(self.base / "profile.json", {})
        if profile:
            ctx += f"USER PROFILE:\n{json.dumps(profile, indent=2)}\n\n"

        goods = self._read_json(self.base / "good_things.json", [])
        if goods:
            chosen = random.sample(goods, min(len(goods), 3))
            ctx += "HAPPY MEMORIES:\n"
            for g in chosen:
                ctx += f"- {g.get('date', '')}: {g.get('memory', '')}\n"
            ctx += "\n"

        events = self._read_json(
            self.base / "emotional_memory" / "emotional_events.json", []
        )
        if events:
            ctx += "RECENT EMOTIONAL EVENTS:\n"
            for e in events[-3:]:
                ctx += f"- {e}\n"
            ctx += "\n"

        return ctx

    def _episodic(self) -> str:
        ctx = "RECENT CONVERSATIONS:\n"
        files = sorted(
            glob.glob(str(self.base / "sessions" / "*.json")), reverse=True
        )
        if not files:
            return ctx + "No previous sessions.\n"
        for fp in files[:3]:
            try:
                data = json.loads(Path(fp).read_text())
                ctx += (
                    f"- [{data.get('timestamp', '?')}] "
                    f"Mode: {data.get('mode', 'unknown')} | "
                    f"Emotion: {data.get('emotion', '?')} | "
                    f"{data.get('summary', '')}\n"
                )
            except Exception:
                pass
        return ctx + "\n"

    def _concept(self) -> str:
        ctx = ""
        concept_dir = self.base / "concept_memory"
        for fname in [
            "robotics_notes.json", "control_theory.json",
            "coding_patterns.json", "project_history.json"
        ]:
            data = self._read_json(concept_dir / fname, None)
            if data:
                label = fname.replace(".json", "").upper()
                ctx += f"{label}:\n{json.dumps(data, indent=2)}\n\n"
        return ctx

    def _creative(self) -> str:
        ctx = ""
        creative_dir = self.base / "creative_memory"
        for fname in [
            "drawings_completed.json", "stories_created.json",
            "user_likes.json", "style_preferences.json"
        ]:
            data = self._read_json(creative_dir / fname, None)
            if data:
                label = fname.replace(".json", "").upper()
                ctx += f"{label}:\n{json.dumps(data, indent=2)}\n\n"
        return ctx

    def get(self, mode: str) -> str:
        if mode == self._cache_mode:
            return self._cache.get(mode, "")

        ctx = "\n[MEMORY CONTEXT]\n"
        if mode == MODE_ALPHA:
            ctx += self._emotional() + self._episodic()
        elif mode == MODE_BETA:
            ctx += self._concept()
            # Brief profile so Beta still knows who Elida is
            profile = self._read_json(self.base / "profile.json", {})
            if profile:
                ctx += f"USER PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
        elif mode == MODE_THETA:
            ctx += self._creative() + self._episodic()
        else:
            ctx += self._emotional() + self._episodic()

        self._cache[mode] = ctx
        self._cache_mode = mode
        return ctx

    def invalidate(self):
        self._cache.clear()
        self._cache_mode = None


# ============================================================================
# CARA MIND — STATE CLASSIFIER & MODE MANAGER
# ============================================================================

class CaraMind:
    """
    Mind-state switchboard.
    Classifies each turn into a mode using sensor fusion + confidence thresholds,
    then routes memory and builds the mode-injected message prefix.
    """

    def __init__(self, memory: CaraMemoryRouter):
        self.memory = memory
        self.mode = MODE_BETA
        self.mode_history: list[dict] = []

    def _parse_visual(self, visual_str: str) -> tuple[str, float]:
        """Parses 'happy (0.85)' → ('happy', 0.85)."""
        try:
            label = visual_str.split("(")[0].strip().lower()
            conf  = float(visual_str.split("(")[1].rstrip(")").strip())
            return label, conf
        except Exception:
            return visual_str.lower().strip(), 0.5

    def _keyword_score(self, text: str, keywords: list[str]) -> float:
        t = text.lower()
        hits = sum(1 for kw in keywords if kw in t)
        return min(hits / max(len(keywords), 1) * 3.0, 1.0)  # scale up sparse hits

    def classify(self, visual_str: str, user_text: str) -> str:
        label, vis_conf = self._parse_visual(visual_str)

        alpha = beta = theta = 0.0

        # Visual emotion signal → Alpha
        if label in ALPHA_EMOTIONS:
            alpha = max(alpha, vis_conf)

        # Keyword signals
        alpha = max(alpha, self._keyword_score(user_text, ALPHA_KEYWORDS) * 0.95)
        beta  = self._keyword_score(user_text, BETA_KEYWORDS)  * 0.90
        theta = self._keyword_score(user_text, THETA_KEYWORDS) * 0.90

        if alpha >= ALPHA_THRESHOLD:
            new_mode = MODE_ALPHA
        elif theta >= THETA_THRESHOLD:
            new_mode = MODE_THETA
        elif beta >= BETA_THRESHOLD:
            new_mode = MODE_BETA
        else:
            new_mode = MODE_NEUTRAL

        if new_mode != self.mode:
            print(
                f"[Mind] {self.mode} → {new_mode}  "
                f"(α={alpha:.2f} β={beta:.2f} θ={theta:.2f})"
            )
            self.memory.invalidate()

        self.mode = new_mode
        self.mode_history.append({"mode": new_mode, "ts": datetime.now().isoformat()})
        return new_mode

    def build_turn_prefix(self, mode: str, visual_str: str) -> str:
        """Returns the block prepended to every user message before sending to Gemini."""
        mem_ctx  = self.memory.get(mode)
        mode_hdr = MODE_PROMPTS[mode]
        visual   = f"[Sensor: visual emotion = {visual_str}]\n"
        return mode_hdr + visual + mem_ctx


# ============================================================================
# SAFETY SUPERVISOR  — runs before every response, above all modes
# ============================================================================

class CaraSafety:
    _DISTRESS = [
        "hurt myself", "end it", "don't want to be here",
        "kill myself", "self-harm", "suicide", "no reason to live",
        "want to die",
    ]
    _CRISIS = (
        "I hear you, and you matter deeply — to me and to others. "
        "Please reach out to someone who can help right now: "
        "a trusted friend, family member, or a crisis line (988 in the US). "
        "I'm right here with you."
    )

    def check(self, text: str) -> tuple[bool, str | None]:
        """Returns (safe, override). If not safe, speak override instead."""
        t = text.lower()
        if any(phrase in t for phrase in self._DISTRESS):
            return False, self._CRISIS
        return True, None


# ============================================================================
# ROS 2 BACKGROUND LISTENER
# ============================================================================

current_visual_emotion = "neutral (0.5)"

class VisualEmotionListener(Node):
    def __init__(self):
        super().__init__("cara_brain_listener")
        self.sub = self.create_subscription(
            String, "/cara/emotion", self._cb, 10
        )
        self.pub_train = self.create_publisher(String, "/cara/feedback", 10)
        self.pub_mode  = self.create_publisher(String, "/cara/mind_mode", 10)

    def _cb(self, msg):
        global current_visual_emotion
        current_visual_emotion = msg.data

    def send_correction(self, true_emotion: str):
        msg = String()
        msg.data = f"SAVE:{true_emotion}"
        self.pub_train.publish(msg)
        self.get_logger().info(f"Training correction sent: {true_emotion}")

    def publish_mode(self, mode: str):
        self.pub_mode.publish(String(data=mode))


def start_ros_thread():
    rclpy.init(args=None)
    node = VisualEmotionListener()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    return node, executor


# ============================================================================
# MEMORY UTILITIES
# ============================================================================

_SAVE_TRIGGERS = [
    "remember that", "write down that", "save this",
    "that was so amazing", "this deserves a happy dance",
    "i am so happy", "i am so blessed", "i am so grateful",
]

def save_happy_memory(user_text: str, base="cara_memory") -> tuple[bool, str | None]:
    content = user_text
    for t in _SAVE_TRIGGERS:
        if t in user_text.lower():
            parts = user_text.lower().split(t, 1)
            if len(parts) > 1:
                content = parts[1].strip().capitalize()
            break

    entry = {"date": datetime.now().strftime("%Y-%m-%d"), "memory": content}
    fp = Path(base) / "good_things.json"
    try:
        data = json.loads(fp.read_text()) if fp.exists() else []
        data.append(entry)
        fp.write_text(json.dumps(data, indent=2))
        return True, content
    except Exception as e:
        print(f"[Memory] Save error: {e}")
        return False, None


# ============================================================================
# BOOTSTRAP
# ============================================================================

load_dotenv()
print("SUCCESS: Libraries loaded.")

print("Connecting to Visual System (ROS 2)...")
ros_node, ros_executor = start_ros_thread()

# Gemini
gemini_api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

with open("cara_memory/cara_prompt.json") as f:
    base_instruction = json.load(f)["system_instruction"]

model = genai.GenerativeModel(
    "models/gemini-2.0-flash",   # update if you have gemini-3-flash access
    system_instruction=base_instruction + LEARNING_INSTRUCTION,
)

# Mind stack
memory_router = CaraMemoryRouter("cara_memory")
cara_mind     = CaraMind(memory_router)
safety        = CaraSafety()

chat_session    = model.start_chat(history=[])
conversation_log: list[dict] = []


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print("\nCara is Waking Up...")
    print("-------------------------------------")
    speak_text("I'm here, and I can see you now.")

    while True:
        # ── A. LISTEN ──────────────────────────────────────────────────────
        user_input = recognize_from_microphone()
        if not user_input:
            continue

        print(f"\nElida: {user_input}")
        conversation_log.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })

        # ── B. EXIT ────────────────────────────────────────────────────────
        if any(w in user_input.lower() for w in ["goodbye", "bye cara"]):
            farewell = "Goodbye Elida. I'll be dreaming of bees until you return."
            print(f"Cara: {farewell}")
            speak_text(farewell)
            conversation_log.append({"role": "assistant", "content": farewell})
            break

        # ── C. SAFETY CHECK (above all modes) ─────────────────────────────
        is_safe, override = safety.check(user_input)
        if not is_safe:
            print(f"[Safety] Override triggered.")
            speak_text(override)
            conversation_log.append({
                "role": "assistant",
                "content": override,
                "safety_triggered": True,
                "timestamp": datetime.now().isoformat(),
            })
            continue

        # ── D. HAPPY MEMORY SAVE ───────────────────────────────────────────
        if any(t in user_input.lower() for t in ["remember that", "save this"]):
            ok, saved = save_happy_memory(user_input)
            if ok:
                print(f" >> Joy Jar updated: {saved}")
                memory_router.invalidate()   # refresh cache so next Alpha turn sees it
                user_input += (
                    "\n[SYSTEM: You just saved this to your permanent 'Joy Jar'. "
                    "Confirm to Elida warmly.]"
                )

        # ── E. CLASSIFY MIND-STATE ─────────────────────────────────────────
        mode = cara_mind.classify(current_visual_emotion, user_input)
        print(f" >> Mode: {mode}  |  Visual: {current_visual_emotion}")

        # ── F. BUILD PROMPT ────────────────────────────────────────────────
        prefix = cara_mind.build_turn_prefix(mode, current_visual_emotion)
        full_message = f"{prefix}\nUser message: {user_input}"

        # ── G. CALL GEMINI ─────────────────────────────────────────────────
        try:
            response   = chat_session.send_message(full_message)
            cara_reply = response.text.strip()
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            cara_reply = "Oh dear, my thoughts got a little tangled. Can you say that again?"

        # ── H. LEARNING TAG ────────────────────────────────────────────────
        if "[LEARN:" in cara_reply:
            try:
                start      = cara_reply.find("[LEARN:") + 7
                end        = cara_reply.find("]", start)
                correction = cara_reply[start:end].strip().lower()
                print(f" >> Learning: retraining on '{correction}'")
                ros_node.send_correction(correction)
                cara_reply = (
                    cara_reply
                    .replace(f"[LEARN: {correction}]", "")
                    .replace(f"[LEARN:{correction}]", "")
                    .strip()
                )
            except Exception as e:
                print(f"[Learn] Parse error: {e}")

        # ── I. OUTPUT ──────────────────────────────────────────────────────
        print(f"Cara [{mode}]: {cara_reply}")
        speak_text(cara_reply)
        conversation_log.append({
            "role": "assistant",
            "content": cara_reply,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        })

        # ── J. PUBLISH MIND-MODE → behavior_node owns servo/blink from here ──
        ros_node.publish_mode(mode)
        print(f"[ROS] Published mind_mode: {mode}")

    # ── CLEANUP ────────────────────────────────────────────────────────────
    print("\n[Saving session memory...]")
    try:
        saved_path = summarize_conversation_for_memory(conversation_log)
        print(f"Session saved → {saved_path}")
    except Exception as e:
        print(f"[Memory] Failed to save: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nForce stopping...")
    finally:
        print("Shutting down ROS...")
        rclpy.shutdown()
