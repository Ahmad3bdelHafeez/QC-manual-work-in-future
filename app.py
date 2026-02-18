import os

from flask import Flask, request, jsonify, render_template
from docx import Document
from docx.shared import Pt
from mistralai import Mistral

from flask_cors import CORS
from flask import send_from_directory

from anthropic import Anthropic
import json
import time
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv


import base64
import io
import json
import asyncio
import os
import time
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from mcp import ClientSession
from mcp.client.sse import sse_client
from mistralai import Mistral
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # allow all origins
ANTHROPIC_API_KEY = "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" #"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"
client = Anthropic(api_key=ANTHROPIC_API_KEY)
MCP_SERVER_URL = "https://playwright-mcp-s8mf.onrender.com/sse"

MCP_URL       = "https://playwright-mcp-s8mf.onrender.com/sse"
MISTRAL_KEY   = "jb0XySiEnvm0r7R3HwSAWvp0aIi80K1v" #"dGA15ms96fjMa6vGmXO7veWlAKInqWVm"
MISTRAL_MODEL = "mistral-large-latest"

MAX_TOOL_RESULT_CHARS = 3000   # truncate huge snapshots to save tokens



MAX_TOOL_RESULT_CHARS = 3000
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)


def mistral_call_with_retry(client, **kwargs):
    """Call Mistral with exponential backoff on rate limit errors."""
    for attempt in range(5):
        try:
            return client.chat.complete(**kwargs)
        except Exception as e:
            if "rate_limited" in str(e) or "1300" in str(e):
                wait = 2 ** attempt
                print(f"Rate limited — retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Mistral rate limit exceeded after retries.")


async def take_screenshot(session, step_num):
    try:
        shot = await session.call_tool("browser_take_screenshot", {"type": "png"})
        for block in shot.content:
            if type(block).__name__ == "ImageContent" and hasattr(block, "data") and block.data:
                # Save first screenshot to disk so we can inspect it
                if step_num == 1:
                    with open("debug_screenshot.png", "wb") as f:
                        f.write(base64.b64decode(block.data))
                    print(f"  → saved debug_screenshot.png")
                return block.data
    except Exception as e:
        print(f"  ✗ screenshot failed: {e}")
    return None

def decode_frame(b64_data):
    """Decode base64 PNG/JPEG into an OpenCV frame, return None on failure."""
    try:
        raw   = base64.b64decode(b64_data)
        arr   = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            print(f"  ✓ decoded frame {frame.shape}")
        else:
            print(f"  ✗ cv2.imdecode returned None (bad image data?)")
        return frame
    except Exception as e:
        print(f"  ✗ decode_frame error: {e}")
        return None


def build_video(steps, video_path, fps=1):
    frames = []
    for step in steps:
        if not step.get("screenshot"):
            print(f"[video] step {step['step_num']} has no screenshot, skipping")
            continue
        frame = decode_frame(step["screenshot"])
        if frame is None:
            continue

        frame = cv2.resize(frame, (1280, 720))

        # Overlay banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (1280, 48), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        label = f"Step {step['step_num']}  |  {step['name']}  |  {json.dumps(step['args'])}"[:110]
        cv2.putText(frame, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 230, 150), 1, cv2.LINE_AA)

        frames.append(frame)

    print(f"[video] total usable frames: {len(frames)}")
    if not frames:
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(video_path, fourcc, fps, (1280, 720))
    for frame in frames:
        for _ in range(max(1, 2 * fps)):
            out.write(frame)
    out.release()

    size = os.path.getsize(video_path)
    print(f"[video] written to {video_path} ({size} bytes)")
    return size > 1000  # sanity check — empty videos are ~1KB


async def run_agent(messages):
    client = Mistral(api_key=MISTRAL_KEY)
    steps  = []

    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema or {},
                    },
                }
                for t in mcp_tools.tools
            ]

            for _ in range(10):
                response = mistral_call_with_retry(
                    client,
                    model=MISTRAL_MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                msg = response.choices[0].message

                if not msg.tool_calls:
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    return msg.content, steps

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    args = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    )

                    result = await session.call_tool(tc.function.name, args)
                    result_text = "\n".join(
                        b.text if hasattr(b, "text") else str(b)
                        for b in result.content
                    )
                    if len(result_text) > MAX_TOOL_RESULT_CHARS:
                        result_text = result_text[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"

                    # Take screenshot
                    screenshot_b64 = await take_screenshot(session, len(steps) + 1)

                    steps.append({
                        "step_num":   len(steps) + 1,
                        "name":       tc.function.name,
                        "args":       args,
                        "result":     result_text,
                        "screenshot": screenshot_b64,
                    })

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "name":         tc.function.name,
                        "content":      result_text,
                    })

            return "Max turns reached.", steps


@app.post("/execute_mistral")
def execute_mistral():
    body     = request.get_json()
    messages = body.get("messages", [])

    if not messages:
        return jsonify({"error": "messages required"}), 400
    if not MISTRAL_KEY:
        return jsonify({"error": "MISTRAL_API_KEY not set"}), 500

    answer, steps = asyncio.run(run_agent(messages))

    # Build video
    video_name = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    video_path = os.path.join(VIDEO_DIR, video_name)
    has_video  = build_video(steps, video_path)

    # Strip screenshots from JSON response (they're in the video)
    clean_steps = [
        {"step": s["step_num"], "tool": s["name"], "args": s["args"], "result": s["result"]}
        for s in steps
    ]

    return jsonify({
        "answer": answer,
        "steps":  clean_steps,
        "video":  f"/video/{video_name}" if has_video else None,
    })


@app.get("/video/<filename>")
def get_video(filename):
    path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="video/mp4")


def run_browser_task(message: str) -> dict:
    """
    Takes a plain English message, sends it to Claude with Playwright MCP,
    and returns the result.
    """
    try:
        print(f"🚀 Running task: {message}")

        response = client.beta.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            betas=["mcp-client-2025-04-04"],
            mcp_servers=[
                {
                    "type": "url",
                    "url": MCP_SERVER_URL,
                    "name": "playwright"
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        # Extract text result
        result_text = "\n".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

        # Extract tools used
        tools_used = [
            {
                "tool": block.name,
                "input": block.input
            }
            for block in response.content
            if block.type == "tool_use"
        ]

        return {
            "success": True,
            "result": result_text,
            "tools_used": tools_used,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@app.route("/execute_claude", methods=["POST"])
def run_execution_task_claude():
    """
    POST /run
    Body: { "message": "go to google.com and search for..." }
    """
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({
            "success": False,
            "error": "Request body must include a 'message' field"
        }), 400

    message = data["message"].strip()

    if not message:
        return jsonify({
            "success": False,
            "error": "'message' cannot be empty"
        }), 400

    result = run_browser_task(message)

    status_code = 200 if result["success"] else 500
    return jsonify(result), status_code


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory("/tmp", filename, as_attachment=True)

def call_mistral_model(prompt):
    client = Mistral(api_key="dGA15ms96fjMa6vGmXO7veWlAKInqWVm")
    model = "mistral-large-2411"
    chat_response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    if chat_response:
        return chat_response.choices[0].message.content.strip()
    else:
        raise Exception("Error calling Mistral model: No response received")

def extract_sections_from_template(template_path):
    document = Document(template_path)
    sections = {}
    section_title = None
    for paragraph in document.paragraphs:
        if paragraph.style.name.startswith("Heading"):
            section_title = paragraph.text.strip()
            sections[section_title] = ""
        elif section_title:
            sections[section_title] += paragraph.text.strip() + " "
    return sections

def fill_sections_with_scope(sections, job_scope):
    filled_sections = {}
    for section, description in sections.items():
        if description.strip():
            prompt = f"Based on the job scope: {job_scope}, fill the section '{section}' with relevant content. Description: {description.strip()}"
            filled_sections[section] = call_mistral_model(prompt)
        else:
            filled_sections[section] = "No description provided."
    return filled_sections

# def update_template_with_filled_sections(template_path, filled_sections, output_path):
#     document = Document(template_path)
#     section_title = None
#     for paragraph in document.paragraphs:
#         if paragraph.style.name.startswith("Heading"):
#             section_title = paragraph.text.strip()
#         elif section_title and section_title in filled_sections:
#             paragraph.text = filled_sections[section_title]
#             section_title = None
#     document.save(output_path)
#     return output_path
from docx import Document
import re

def insert_markdown(paragraph, markdown_text):
    """
    Handles basic Markdown: bold (**text**), italic (*text*), and headings (#, ##, ###).
    """
    # Clear existing text
    paragraph.clear()

    # Check if the line is a heading
    # heading_match = re.match(r'^(#{1,2})\s+(.*)', markdown_text)
    # if heading_match:
    #     hashes, heading_text = heading_match.groups()
    #     level = len(hashes)  # number of # determines heading level
    #     paragraph.style = f"Heading {level}"
    #     paragraph.add_run(heading_text)
    #     return

    # Otherwise, parse inline bold/italic
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*)', markdown_text)
    for token in tokens:
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token.strip("*"))
            run.bold = True
        elif token.startswith("*") and token.endswith("*"):
            run = paragraph.add_run(token.strip("*"))
            run.italic = True
        else:
            paragraph.add_run(token)

def update_template_with_filled_sections(template_path, filled_sections, output_path):
    document = Document(template_path)
    section_title = None

    for paragraph in document.paragraphs:
        if paragraph.style.name.startswith("Heading"):
            section_title = paragraph.text.strip()
        elif section_title and section_title in filled_sections:
            # Clear paragraph
            paragraph.text = ""
            # Insert formatted Markdown
            insert_markdown(paragraph, filled_sections[section_title])
            section_title = None

    document.save(output_path)
    return output_path


@app.route('/generate-test-plan', methods=['POST'])
def generate_test_plan():
    data = request.json
    job_scope = data.get('job_scope')
    template_path = os.path.join(os.path.dirname(__file__), "templates", "test_plan_template.docx")
    output_path = os.path.join("/tmp", "Updated_Test_Plan.docx")

    if not job_scope:
        return jsonify({"error": "job_scope is required"}), 400
    try:
        sections = extract_sections_from_template(template_path)
        filled_sections = fill_sections_with_scope(sections, job_scope)
        result_path = update_template_with_filled_sections(template_path, filled_sections, output_path)
        print(jsonify({"message": "Test plan generated", "output_path": result_path}))
        return jsonify({"message": "Test plan generated", "output_path": result_path})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)})

@app.route('/generate-test-cases', methods=['POST'])
def generate_test_cases():
    data = request.json
    user_story = data.get('user_story')
    print(user_story)

    if not user_story:
        return jsonify({"error": "user_story is required"}), 400

    prompt = f"{user_story}, from the given user story create a table of test cases (Test Case ID, Description (starts with validate that....), Steps, Expected Behavior) with corner cases, give me the table only"
    result = call_mistral_model(prompt)
    print(result)
    print({"message": "Test Cases generated", "output": result})
    return jsonify({"message": "Test Cases generated", "output": result})

@app.route('/generate-bug-report', methods=['POST'])
def generate_bug_report():
    data = request.json
    bugs = data.get('bugs')
    print(bugs)

    if not bugs:
        return jsonify({"error": "bugs is required"}), 400

    prompt = f"{bugs}, from the given bugs, create a bug report as a table with all attributes (Bug ID,Description,Steps to Reproduce,Expected Behavior,Actual Behavior,Priority,Severity,Status), give me the table only"
    result = call_mistral_model(prompt)
    print(result)
    print({"message": "Bug report generated", "output": result})
    return jsonify({"message": "Bug report generated", "output": result})

@app.route('/generate-data', methods=['POST'])
def generate_data():
    data = request.json
    input = data.get('input')
    print(input)

    if not input:
        return jsonify({"error": "input is required"}), 400

    prompt = f"""You are a dummy data generator.  
I will provide you with an input description.  
You must always produce only a table as output, with exactly the column names and format I specify in the description.  

Rules:  
- Output must be a table only (no explanations, no extra text).  
- Column headers must exactly match what I specify.  
- Each row must contain one complete and realistic data record.  
- Number of rows = as specified in my input.  
- Data must look realistic (not just random gibberish).  
- Ensure uniqueness where it makes sense (e.g., IDs, emails).  

Example inputs:  
1. "10 questions about employees, their salaries and departments → table with 2 columns (question #, question)"  
2. "emails and passwords → table with 2 columns (email, password), 15 rows"  
3. "product codes and prices → table with 2 columns (product_code, price), 20 rows"  

Now wait for my input and generate the table accordingly. 

input description: {input}"""
    result = call_mistral_model(prompt)
    print(result)
    print({"message": "Dummy data generated", "output": result})
    return jsonify({"message": "Dummy data generated", "output": result})

@app.route('/generate-data-quality-tests', methods=['POST'])
def generate_data_quality_tests():
    data = request.json
    input = data.get('schema')
    print(input)

    if not input:
        return jsonify({"error": "schema is required"}), 400

    prompt = f"""You are a mySQL data quality and testing expert.
    Given a table schema, table name, and list of columns with their data types, generate a comprehensive SQL validation script that covers both schema tests and data quality tests, returning a unified result set with the following columns:
    Check_Type | Dimension | Rule | Expectation | Failed_Records

    Check_Type (e.g., Schema, Data,...)
    Dimension must be in (e.g., Completeness, Integrity, Uniqueness, Validity, Consistency,...)
    Rule (e.g. Invalid mental_health_rating, Attendance<50 but Exam>95,...)
    Expectation (e.g. Diet quality must be Poor, Average, or Good, If sleep_hours<3, mental_health_rating should not exceed 9,...)
    Failed_Records (e.g. number of failed records)

    Your task:

    Schema Tests:

    - Row count test → Verify expected record count (use a placeholder number like 2000).

    - Column count test → Verify number of columns matches the provided schema.

    - Column names and types test → Verify each column matches its expected data type.

    Data Quality Tests:

    - Uniqueness: Check if key columns (like IDs) are unique.

    - Completeness: Check for nulls in critical fields (IDs, required attributes, metrics).

    - Validity: Based on data type and semantic meaning, create range and categorical validation rules:

    - For numeric columns: specify realistic ranges (e.g., 0–100 for percentages or scores).

    - For categorical columns: specify allowed values (e.g., Gender: Male/Female/Other).

    - Consistency: Create logical dependency rules (e.g., if attendance < 50, exam_score < 95).

    - Use UNION ALL between each test block.

    - Include descriptive comments (e.g., -- =======================).

    - Use the given schema and table name consistently.

    Input Example:
    
    {input}


    Output Format Should be the script only, Return a single SQL script that includes:

    - Clear comment sections (-- SCHEMA TESTS, -- DATA QUALITY TESTS)

    - A combined query using UNION ALL

    - Proper CASE logic for validation rules

    - Friendly readable expectations (e.g., 'Attendance must be between 0 and 100')

    The final output should look similar in structure to the following:
    -- =======================
    -- SCHEMA TESTS
    -- =======================
    SELECT ...
    UNION ALL
    ...
    -- =======================
    -- DATA QUALITY TESTS
    -- =======================
    SELECT ...
    """

    print(f'#### {prompt}')
    result = call_mistral_model(prompt)
    print(result)
    print({"message": "Data Quality Tests generated", "output": result})
    return jsonify({"message": "Data Quality Tests generated", "output": result})
# Serve home page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)


























