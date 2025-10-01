import os

from flask import Flask, request, jsonify
from docx import Document
from mistralai import Mistral

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow all origins

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

    sections = extract_sections_from_template(template_path)
    filled_sections = fill_sections_with_scope(sections, job_scope)
    result_path = update_template_with_filled_sections(template_path, filled_sections, output_path)
    print(jsonify({"message": "Test plan generated", "output_path": result_path}))
    return jsonify({"message": "Test plan generated", "output_path": result_path})

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

if __name__ == "__main__":
    app.run(debug=True)
