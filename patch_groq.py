import urllib.request
import os

with open("app.py", "r") as f:
    app_code = f.read()
    
# Replace getenv
replacement = """
groq_key = os.getenv("GROQ_API_KEY")
try:
    if not groq_key and hasattr(st, "secrets"):
        groq_key = st.secrets.get("GROQ_API_KEY")
except:
    pass
client = Groq(api_key=groq_key)
"""

app_code = app_code.replace('client = Groq(api_key=os.getenv("GROQ_API_KEY"))', replacement)
with open("app.py", "w") as f:
    f.write(app_code)


with open("src/agentic_rag/workflow.py", "r") as f:
    wf_code = f.read()

wf_replacement = """
        import streamlit as st
        groq_key = os.getenv("GROQ_API_KEY")
        try:
            if not groq_key and hasattr(st, "secrets"):
                groq_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
        client = Groq(api_key=groq_key)
"""
wf_code = wf_code.replace('client = Groq(api_key=os.getenv("GROQ_API_KEY"))', wf_replacement)
with open("src/agentic_rag/workflow.py", "w") as f:
    f.write(wf_code)
