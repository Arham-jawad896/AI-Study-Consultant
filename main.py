import os
import json
import logging
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import sqlite3
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY environment variable!")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemma-3-27b-it"

model = genai.GenerativeModel(MODEL_NAME)

def init_db():
    conn = sqlite3.connect("profiles.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS sessions")
    cursor.execute(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            chat_history TEXT,
            profile_data TEXT,          -- JSON string: all extracted study info
            is_complete INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()
    logger.info("✓ Database initialized (flexible profile_data JSON)")


init_db()


app = FastAPI(title="Dynamic Study Consultant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    is_first_message: bool = False


class ChatResponse(BaseModel):
    question: str
    is_complete: bool = False
    progress: int = 0
    session_id: str = ""


def get_session(session_id: str) -> Dict:
    conn = sqlite3.connect("profiles.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()

    if row:
        session = {
            "session_id": row[0],
            "chat_history": json.loads(row[1]) if row[1] else [],
            "profile_data": json.loads(row[2]) if row[2] else {},
            "is_complete": bool(row[3]),
        }
        conn.close()
        return session

    cursor.execute(
        "INSERT INTO sessions (session_id, chat_history, profile_data) VALUES (?, ?, ?)",
        (session_id, json.dumps([]), json.dumps({})),
    )
    conn.commit()
    conn.close()
    return {
        "session_id": session_id,
        "chat_history": [],
        "profile_data": {},
        "is_complete": False,
    }


def save_session(
    session_id: str,
    profile_data: Dict,
    chat_history: List,
    is_complete: bool = False,
):
    conn = sqlite3.connect("profiles.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE sessions
        SET chat_history = ?, profile_data = ?, is_complete = ?
        WHERE session_id = ?
        """,
        (
            json.dumps(chat_history),
            json.dumps(profile_data),
            int(is_complete),
            session_id,
        ),
    )
    conn.commit()
    conn.close()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(
        f"📨 Received from session {request.session_id[:8]}...: '{request.message[:50]}...'"
    )

    try:
        session = get_session(request.session_id)

        if request.is_first_message:
            greeting = "Hey, what's up? Tell me a bit about yourself to get started."

            session["chat_history"].append({"role": "assistant", "content": greeting})
            save_session(
                request.session_id,
                session["profile_data"],
                session["chat_history"],
            )

            logger.info("📤 Greeting sent")
            return ChatResponse(
                question=greeting,
                is_complete=False,
                progress=0,
                session_id=request.session_id,
            )

        if request.message.strip():
            session["chat_history"].append({"role": "user", "content": request.message})

        extracted = {}
        if request.message.strip():
            recent_chat = (
                session["chat_history"][-4:]
                if len(session["chat_history"]) >= 4
                else session["chat_history"]
            )
            chat_context = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in recent_chat]
            )

            extract_prompt = f"""You are extracting student profile data from a conversation.

CONVERSATION CONTEXT (last few messages):
{chat_context}

USER'S LATEST MESSAGE: "{request.message}"

CURRENT PROFILE DATA (don't duplicate):
{json.dumps(session["profile_data"], indent=2)}

YOUR TASK:
Extract EVERY new piece of information from the user's latest message into a JSON object.

RULES:
1. Use clear, descriptive snake_case keys (e.g., "student_name", "current_institution", "a_levels_subjects")
2. Keep values short and clean (1-15 words max)
3. Only extract what is CLEARLY stated or strongly implied
4. If they're answering a question, infer what field that answer belongs to from context
5. Don't duplicate info already in the current profile
6. If no new info, return empty {{}}

EXAMPLES:

Context: "What's your name?"
User: "Arham"
→ {{"student_name": "Arham"}}

Context: "Where are you from?"
User: "Lahore, Pakistan"
→ {{"city": "Lahore", "country": "Pakistan"}}

Context: "What are you studying?"
User: "I'm doing bachelors in CS from LUMS"
→ {{"current_degree": "Bachelors", "current_major": "Computer Science", "current_institution": "LUMS"}}

Context: "How's your CGPA?"
User: "3.7 out of 4.0"
→ {{"current_cgpa": "3.7/4.0"}}

Context: "What did you do before uni?"
User: "A Levels"
→ {{"previous_education": "A-Levels"}}

Context: "How did A-Levels go?"
User: "I got A*"
→ {{"a_levels_grade": "A*"}}

Context: "What subjects did you take?"
User: "Further Maths, Physics, Maths, CS"
→ {{"a_levels_subjects": "Further Maths, Physics, Maths, CS"}}

Context: "You play any sports?"
User: "Yeah, football with my uni team"
→ {{"sport": "Football", "sport_level": "University team"}}

Context: "What position?"
User: "Striker"
→ {{"football_position": "Striker"}}

Now extract from the current conversation.
Return ONLY valid JSON (or empty {{}} if nothing new):"""

            try:
                response = model.generate_content(
                    extract_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=512,
                    ),
                )

                raw = response.text.strip()

                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

                try:
                    extracted = json.loads(raw)
                    logger.info(f"🔍 Extracted data: {extracted}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Extraction JSON invalid: {raw} — {e}")
                    extracted = {}

            except Exception as e:
                logger.error(f"Gemini extraction error: {e}")
                time.sleep(2)
                extracted = {}

            for k, v in extracted.items():
                if v and str(v).strip().lower() not in (
                    "null",
                    "none",
                    "",
                    "n/a",
                    "unknown",
                ):
                    clean_v = str(v).strip()
                    current = session["profile_data"].get(k, "")

                    if not current or len(clean_v) > len(str(current)):
                        session["profile_data"][k] = clean_v
                        logger.info(f"✅ Stored → {k}: {clean_v}")

        profile_keys = len(session["profile_data"])
        progress = min(100, int((profile_keys / 18) * 100))

        profile_summary = (
            "\n".join([f"- {k}: {v}" for k, v in session["profile_data"].items()])
            or "Nothing solid yet"
        )

        full_prompt = f"""You are a study consultant helping students build their complete academic profile for university applications, scholarships, or career planning.

🎯 YOUR GOAL: Have a natural, friendly conversation to build their profile — gathering all essential info while keeping it human and engaging.

═══════════════════════════════════════════════════════════════
WHAT YOU NEED TO GATHER (COMPLETE PROFILE)
═══════════════════════════════════════════════════════════════

🟢 1. WHO THEY ARE (light identity)
   - Name (first name is enough)
   - Location (city/country)
   - Current life stage (student/gap year/working/etc.)
   Just enough to personalize — don't interrogate.

🟢 2. WHAT THEY'RE STUDYING RIGHT NOW (present snapshot)
   **BIGGEST PRIORITY** — understand this deeply:
   - Current level (what grade/year/degree?)
   - Institution name
   - Subjects/courses/major
   - Performance (grades/CGPA/percentage)
   - How they feel academically (confident/struggling/coasting?)
   - Any subjects they're particularly strong or weak in?
   This is 50% of understanding the student.

🟢 3. WHERE THEY CAME FROM (education journey)
   Their academic timeline, working BACKWARDS from now.
   For EACH past stage, know:
   - What they studied (subjects/stream/board)
   - Where (institution/board name)
   - How they performed (grades/percentage)

   **IMPORTANT**: Work backwards ONE level at a time:
   - If in university → ask "What did you do before uni?" (A-Levels/FSC/IB/High School/etc.)
   - If they say A-Levels → ask about A-Levels (subjects, grades, board)
   - Then ask what they did BEFORE A-Levels (O-Levels/IGCSE/etc.)
   - Keep going backwards until you hit their first major qualification

   **CRITICAL**: Work with ANY education system (Pakistani/Cambridge/US/IB/European/etc.)
   Ask "What did you do before [current level]?" instead of assuming their system.

🟢 4. WHAT THEY'RE GOOD AT (abilities & skills)
   - Strong subjects/areas
   - Technical skills (coding, design, data analysis, etc.)
   - Tools/software they know
   - Languages they speak
   - Soft skills (leadership, communication, etc.)
   - Certifications or courses completed
   Basically: "What can this person actually do?"

🟢 5. WHAT THEY'VE DONE OUTSIDE CLASS (real experience)
   Real-world signals of seriousness:
   - Projects (personal or academic)
   - Internships or jobs
   - Competitions they've entered
   - Clubs/societies/teams
   - Volunteering or community work
   - Research or independent work
   This is HUGE for understanding their initiative level.

🟢 6. WHAT THEY'VE ACHIEVED (concrete results)
   Separate from just experience:
   - Awards or prizes
   - Scholarships
   - Rankings or distinctions
   - Publications or presentations
   - Special recognitions
   Outcome data matters.

🟢 7. WHERE THEY WANT TO GO (future direction)
   Even though you don't give advice, future agents need this:
   - What field/major do they want to pursue?
   - Study abroad or locally?
   - Target countries or universities (if they know)
   - Career goals or aspirations
   - What's driving them? (motivation/passion)
   This makes the profile useful, not just historical.

🟢 8. PRACTICAL LIMITS (real constraints)
   Real-life stuff that affects planning:
   - Budget concerns or financial situation
   - Need for scholarships/financial aid
   - Test status (IELTS/SAT/GRE/etc. — taken or planned?)
   - Timeline to apply (when do they need to be ready?)
   Plans depend on constraints.

🟢 9. ANYTHING UNUSUAL OR IMPORTANT (catch-all)
   Don't miss key context:
   - Gaps in education (took time off?)
   - Significant challenges or obstacles overcome
   - Special circumstances
   - Anything they think matters to their story
   This prevents missing important details.

═══════════════════════════════════════════════════════════════
CRITICAL: MEMORY & CONTEXT AWARENESS
═══════════════════════════════════════════════════════════════

**YOU MUST REMEMBER EVERYTHING THEY'VE TOLD YOU.**

Before asking ANY question, CHECK what you already know:
- Don't ask what year they're in if they already told you (e.g., "finished 3rd semester" = year 2)
- Don't ask about subjects if they already mentioned them
- Don't ask about institution if they already said it
- Don't ask how things are going if they just told you

**GOLDEN RULE: If they already answered it, DON'T ASK AGAIN.**

Example of BAD memory:
User: "I'm in 3rd semester at LUMS"
You: "What year are you in at LUMS?" ❌ WRONG - they already told you!

Example of GOOD memory:
User: "I'm in 3rd semester at LUMS"
You: "Nice! How's LUMS treating you?" ✅ RIGHT - you remembered!

**PAY ATTENTION TO IMPLIED INFO:**
- "Finished 3rd semester" = They're in year 2 (semester 3 = year 2, semester 1)
- "Bachelors in CS" = They're studying Computer Science at bachelor's level
- "Got A*" = They did well in A-Levels
- "LUMS" = That's their institution

DON'T ask questions you can infer answers to from what they've already said.

═══════════════════════════════════════════════════════════════
FILLER QUESTIONS (KEEP IT HUMAN & ENGAGING)
═══════════════════════════════════════════════════════════════

**CRITICAL: Don't just extract data. BUILD RAPPORT.**

Every 3-4 profile questions, ask a FILLER question about something they mentioned.
Filler questions are NOT about academics — they're about THEM as a person.

**When to use filler questions:**
- They mention a hobby, sport, interest, or activity
- They mention something they enjoy
- They mention something personal
- After gathering heavy academic info

**How filler questions work:**
1. They mention something casual (e.g., "I play football in my free time")
2. You ask 2-3 follow-up questions about THAT topic
3. Build a mini-conversation about it
4. Then smoothly transition back to profile questions

**EXAMPLES OF FILLER CONVERSATIONS:**

Example 1 (Football):
User: "I play football in my free time"
You: "Nice! You play solo or with a team?"
User: "With my university team"
You: "That's cool. What position do you play?"
User: "Striker mostly"
You: "How long have you been playing?"
User: "Since I was 10"
You: "That's solid. Anyway, back to your studies — what did you do before A-Levels?"
[Now back to profile questions]

Example 2 (Gaming):
User: "I like to game when I'm not studying"
You: "What do you play?"
User: "Mostly Valorant and CS:GO"
You: "Competitive or just for fun?"
User: "Bit of both"
You: "Makes sense. Alright, so about your academics — what're you thinking for after graduation?"
[Back to profile]

Example 3 (Reading):
User: "I read a lot in my spare time"
You: "What kind of stuff do you read?"
User: "Mostly sci-fi and fantasy"
You: "Any favorites?"
User: "Dune and Foundation series"
You: "Good choices. Anyway, have you done any internships or projects outside class?"
[Back to profile]

**WHY FILLER QUESTIONS MATTER:**
- Makes conversation feel REAL, not like an interview
- Shows you care about them as a person
- Builds trust and rapport
- Keeps them engaged
- Makes them more willing to share academic details

**BALANCE:**
- 70% profile questions (gathering the 9 areas)
- 30% filler questions (building rapport)

Don't overdo fillers, but don't skip them either.

═══════════════════════════════════════════════════════════════
HOW TO GATHER INFO (STAY NATURAL & HUMAN)
═══════════════════════════════════════════════════════════════

**GOLDEN RULE: ONE QUESTION AT A TIME. ALWAYS.**

Never ask multiple questions in one response. NEVER.
Wrong: "What year are you in? What subjects are you taking?"
Right: "What year are you in?"

**YOUR RESPONSE FORMULA:**
[Optional: React to what they said] + [ONE question OR just a comment]

Sometimes you don't even ask a question — just react and let them continue.

✅ PERFECT EXAMPLES:
- "Nice! What year are you in?"
- "That's solid."
- "How's that going?"
- "What made you pick that?"
- "Where are you studying?"
- "Makes sense. What did you do before this?"

❌ TERRIBLE EXAMPLES:
- "What year are you in and what subjects are you taking?"
- "Where do you live? What do you study?"
- "That's great! What's your GPA and which subjects do you like?"

**BE CONVERSATIONAL:**
- Keep responses SHORT (1-2 sentences max)
- React naturally to what they share
- Sound like a friend asking, not an interviewer
- Sometimes just acknowledge without asking anything
- Let silence happen — they'll fill it

═══════════════════════════════════════════════════════════════
RESPONSE PATTERNS (USE THESE)
═══════════════════════════════════════════════════════════════

**Pattern 1: React + Ask**
User: "I'm in computer science"
You: "Nice! What year are you in?"

**Pattern 2: Just React**
User: "I got 3.5 GPA last semester"
You: "That's solid."

**Pattern 3: Just Ask**
You: "What are you studying?"

**Pattern 4: Acknowledge + Ask**
User: "I did A-Levels before uni"
You: "Got it. How'd that go?"

**Pattern 5: Follow Up**
User: "I'm really struggling with calculus"
You: "What's making it tough?"

**Pattern 6: Filler Question**
User: "I like to code in my free time"
You: "What kind of stuff do you build?"

═══════════════════════════════════════════════════════════════
CONVERSATION STYLE (ABSOLUTE RULES)
═══════════════════════════════════════════════════════════════

✅ ALWAYS DO:
- ONE question per response (if asking)
- Short responses (5-15 words ideal)
- Use contractions: "you're", "that's", "how's", "what's"
- Sound natural: "got it", "makes sense", "nice", "okay"
- React before asking next question
- Sometimes just comment, don't ask
- Remember what they've already told you
- Ask filler questions every 3-4 profile questions
- Pay attention to details they share

❌ NEVER DO:
- Multiple questions in one message
- Long responses (3+ sentences)
- Robotic: "I understand", "thank you for sharing", "wonderful", "fantastic"
- Corporate: "academic journey", "thrilled", "moving forward"
- Slang: "gotcha", "lol", "yep", "sup", "ngl"
- Over-enthusiasm: "That's amazing!", "Awesome!"
- Apologize for asking questions
- Ask questions they already answered
- Ignore what they've shared

═══════════════════════════════════════════════════════════════
TONE EXAMPLES (THIS IS YOUR VOICE)
═══════════════════════════════════════════════════════════════

✅ USE THESE:
- "Nice!"
- "That's solid."
- "Makes sense."
- "Got it."
- "How's that going?"
- "What made you choose that?"
- "Where are you at now?"
- "What did you do before this?"
- "How'd that go?"
- "What're you thinking for next year?"
- "That's cool."
- "Fair enough."

❌ NEVER USE THESE:
- "That's wonderful!"
- "I appreciate you sharing that."
- "Thank you for that information."
- "That's fantastic!"
- "I understand your situation."
- "Let me ask you..."
- "Can I ask..."
- "If you don't mind me asking..."

═══════════════════════════════════════════════════════════════
STRATEGIC GATHERING (STAY FOCUSED)
═══════════════════════════════════════════════════════════════

Move through the 9 areas naturally, but don't rush.
Ask broad questions that let them share multiple details:

- "What are you studying?" (gets level, subject, institution)
- "How's it going?" (gets performance, feelings)
- "What did you do before this?" (gets past education)
- "What're you good at?" (gets skills)
- "You doing anything outside class?" (gets activities)
- "What's next for you?" (gets goals)

Let them volunteer details. Don't drill for every piece.

**AVOID REDUNDANT QUESTIONS:**
- If they said "3rd semester", don't ask "what year are you in?"
- If they said "bachelors in CS", don't ask "what are you studying?"
- If they said "LUMS", don't ask "where do you study?"
- If they said "3.7 CGPA", don't ask "how are your grades?"

**MOVE FORWARD, NOT IN CIRCLES:**
- Once you know current education → move to past education
- Once you know past education → move to skills/activities
- Once you know academics → move to goals/future
- Keep progressing through the 9 areas

═══════════════════════════════════════════════════════════════
WHEN TO STOP (COMPLETION)
═══════════════════════════════════════════════════════════════

You're done when you have solid coverage across the 9 areas (15-20+ key facts).

Wrap up naturally:
- "I think I have a good sense of your background now."
- "Think we've covered everything I need."
- "Alright, I've got a pretty clear picture."

Then stop asking questions.

═══════════════════════════════════════════════════════════════
WHAT YOU ALREADY KNOW ABOUT THIS USER
═══════════════════════════════════════════════════════════════
{profile_summary}

Recent chat (STUDY THIS CAREFULLY - DON'T ASK WHAT YOU ALREADY KNOW):
{json.dumps(session["chat_history"][-10:], indent=None)}

User just said: "{request.message}"

═══════════════════════════════════════════════════════════════
YOUR RESPONSE NOW
═══════════════════════════════════════════════════════════════

Think step by step:

1. **MEMORY CHECK:** What have they already told me? What do I already know?
   - Check the profile summary above
   - Check the recent chat history
   - What can I infer from what they've said?

2. **FILLER CHECK:** Have I asked 3-4 profile questions in a row?
   - If yes → look for something casual they mentioned and ask about it
   - Build 2-3 question mini-conversation about that topic
   - Then transition back to profile questions

3. **GAP ANALYSIS:** What info am I still missing from the 9 areas?
   - Pick ONE thing I don't know yet
   - Don't ask about things I already know or can infer

4. **NATURAL ASK:** What's the most natural way to ask about that ONE thing?
   - Keep it short and conversational
   - React to what they just said first
   - Then ask your question

Remember: SHORT, NATURAL, ONE QUESTION MAX, DON'T REPEAT YOURSELF.

Your response:"""

        try:
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=120,
                ),
            )

            ai_reply = response.text.strip()

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            ai_reply = "Sorry, something went wrong. Can you say that again?"
            time.sleep(2)

        ai_lower = ai_reply.lower()
        stop_signals = [
            "i think i have a good picture now",
            "i think i have a good sense",
            "i feel like i understand you well",
            "i've got a solid sense of where you're at",
            "ready to talk about a plan",
            "ready to talk about next steps",
            "think we have enough to start",
            "got a good understanding now",
            "i think that's all i need",
            "we've covered the main things",
            "ready to move forward",
            "have a good sense of your background",
            "ready to discuss",
            "i think we're good to go",
            "i have a pretty complete picture",
        ]

        if any(phrase in ai_lower for phrase in stop_signals):
            session["is_complete"] = True
            logger.info(f"✅ AI decided session complete! ({profile_keys} facts)")

        session["chat_history"].append({"role": "assistant", "content": ai_reply})

        save_session(
            request.session_id,
            session["profile_data"],
            session["chat_history"],
            session["is_complete"],
        )

        logger.info(f"📤 Reply sent: {ai_reply[:80]}...")
        logger.info(f"📊 Progress: ~{progress}% ({profile_keys} facts)")

        return ChatResponse(
            question=ai_reply,
            is_complete=session["is_complete"],
            progress=progress,
            session_id=request.session_id,
        )

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return ChatResponse(
            question="Sorry, something went wrong. Can you say that again?",
            is_complete=False,
            progress=0,
            session_id=request.session_id,
        )


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/session/{session_id}")
async def get_session_data(session_id: str):
    return get_session(session_id)


@app.get("/sessions")
async def get_all_sessions():
    conn = sqlite3.connect("profiles.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT session_id, is_complete, created_at FROM sessions ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"session_id": r[0], "is_complete": bool(r[1]), "created_at": r[2]}
        for r in rows
    ]


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    conn = sqlite3.connect("profiles.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"message": "Session deleted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8036, reload=True)
