import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # We need to find the specific block and indent it properly.
    pattern = re.compile(
        r"^(groq_key = os\.getenv\(\"GROQ_API_KEY\"\)\n"
        r"try:\n"
        r"    if not groq_key and hasattr\(st, \"secrets\"\):\n"
        r"        groq_key = st\.secrets\.get\(\"GROQ_API_KEY\"\)\n"
        r"except:\n"
        r"    pass\n"
        r"client = Groq\(api_key=groq_key\))$",
        re.MULTILINE
    )

    def replacer(match):
        # Determine the indentation by looking at the line before it or hardcoding!
        # If it's app.py, it's indented by 28 spaces.
        # If it's workflow.py, it's indented by 8 spaces.
        pass
    
    # Actually, simpler: Since we know exactly where it is, let's just find the context.
    # In app.py:
    #                         with st.spinner("Thinking..."):
    # groq_key = ...

    content = content.replace(
"""
groq_key = os.getenv("GROQ_API_KEY")
try:
    if not groq_key and hasattr(st, "secrets"):
        groq_key = st.secrets.get("GROQ_API_KEY")
except:
    pass
client = Groq(api_key=groq_key)""",
"""
                            groq_key = os.getenv("GROQ_API_KEY")
                            try:
                                if not groq_key and hasattr(st, "secrets"):
                                    groq_key = st.secrets.get("GROQ_API_KEY")
                            except:
                                pass
                            client = Groq(api_key=groq_key)"""
    )
    with open(filepath, 'w') as f:
        f.write(content)

def fix_workflow():
    with open('src/agentic_rag/workflow.py', 'r') as f:
        content = f.read()
    content = content.replace(
"""
groq_key = os.getenv("GROQ_API_KEY")
try:
    if not groq_key and hasattr(st, "secrets"):
        groq_key = st.secrets.get("GROQ_API_KEY")
except:
    pass
        client = Groq(api_key=groq_key)""",
"""
        groq_key = os.getenv("GROQ_API_KEY")
        try:
            if not groq_key and hasattr(st, "secrets"):
                groq_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
        client = Groq(api_key=groq_key)""")
    with open('src/agentic_rag/workflow.py', 'w') as f:
        f.write(content)

fix_file('app.py')
fix_file('app/streamlit_app.py')
fix_workflow()
print("Fixed!")
