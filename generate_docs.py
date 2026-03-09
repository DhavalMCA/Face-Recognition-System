import os
import ast
from pathlib import Path

# Paths
ROOT_DIR = r"d:\Dhaval Prajapati\Freelancer Project\FewShotFace"
OUTPUT_DIR = os.path.join(ROOT_DIR, "Project Documentation")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Bootstrap_Documentation.html")

# Custom CSS with CSS Variables for Theme
CSS_CONTENT = """
:root {
    /* Color Palette */
    --deep-blue: #0A2540;
    --primary-accent: #0066FF;
    --soft-gray: #F6F9FC;
    --text-main: #32325D;
    --text-muted: #6B7C93;
    --white-bg: #FFFFFF;
    --border-light: #E6EBF1;
    --code-bg: #1A1F36;
    --code-text: #F4F7FA;
    --card-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
    --card-hover-shadow: 0 15px 35px rgba(50, 50, 93, 0.1), 0 5px 15px rgba(0, 0, 0, 0.07);
    --navbar-height: 70px;
    --sidebar-width: 280px;
}

body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    color: var(--text-main);
    background-color: var(--soft-gray);
    padding-top: var(--navbar-height);
    scroll-behavior: smooth;
    position: relative;
}

/* Navbar */
.navbar-custom {
    background-color: var(--white-bg);
    border-bottom: 1px solid var(--border-light);
    height: var(--navbar-height);
    padding: 0 2rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.02);
}
.navbar-brand {
    font-weight: 700;
    color: var(--deep-blue) !important;
    font-size: 1.4rem;
}
.navbar-brand i { color: var(--primary-accent); margin-right: 8px; }

/* Sidebar */
.sidebar {
    background-color: var(--white-bg);
    border-right: 1px solid var(--border-light);
    height: calc(100vh - var(--navbar-height));
    position: sticky;
    top: var(--navbar-height);
    overflow-y: auto;
    padding: 1.5rem 1rem;
    box-shadow: 2px 0 5px rgba(0,0,0,0.02);
    transition: all 0.3s ease;
    z-index: 1000;
}
.sidebar .search-container { margin-bottom: 1.5rem; }
.sidebar .search-box {
    border-radius: 8px;
    background-color: var(--soft-gray);
    border: 1px solid var(--border-light);
    padding: 0.5rem 1rem;
    width: 100%;
    color: var(--text-main);
    transition: all 0.2s;
}
.sidebar .search-box:focus {
    background-color: var(--white-bg);
    border-color: var(--primary-accent);
    box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.15);
    outline: none;
}
.sidebar .nav-group {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05rem;
    font-weight: 700;
    color: var(--text-muted);
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    padding-left: 0.5rem;
}
.sidebar .nav-link {
    color: var(--text-main);
    font-size: 0.9rem;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    transition: all 0.2s ease;
    margin-bottom: 0.2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sidebar .nav-link:hover {
    background-color: var(--soft-gray);
    color: var(--primary-accent);
}
.sidebar .nav-link.active {
    background-color: var(--primary-accent);
    color: var(--white-bg);
    font-weight: 500;
}

/* Main Content */
.main-content {
    padding: 2.5rem;
    max-width: 1100px;
    margin: 0 auto;
}
.section-header {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--border-light);
}
.section-header h1 {
    font-weight: 800;
    color: var(--deep-blue);
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}
.section-header .lead { color: var(--text-muted); font-size: 1.1rem; }

/* Cards for Functions */
.doc-card {
    background-color: var(--white-bg);
    border-radius: 12px;
    border: 1px solid var(--border-light);
    box-shadow: var(--card-shadow);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 3rem;
    overflow: hidden;
    scroll-margin-top: calc(var(--navbar-height) + 2rem);
}
.doc-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--card-hover-shadow);
}
.doc-card-header {
    background: linear-gradient(135deg, var(--white-bg) 0%, var(--soft-gray) 100%);
    padding: 1.5rem 2rem;
    border-bottom: 1px solid var(--border-light);
}
.doc-card-header h2 { margin: 0; font-size: 1.5rem; font-weight: 700; color: var(--deep-blue); display: flex; align-items: center; gap: 10px; }
.doc-card-header .badge { font-weight: 500; padding: 0.4em 0.8em; font-size: 0.75rem; letter-spacing: 0.05em; }
.doc-card-body { padding: 2rem; }

.info-block { display: flex; margin-bottom: 1rem; }
.info-label { width: 140px; font-weight: 600; color: var(--text-muted); flex-shrink: 0; }
.info-desc { flex-grow: 1; color: var(--text-main); }

.purpose-box {
    background-color: rgba(0, 102, 255, 0.05);
    border-left: 4px solid var(--primary-accent);
    padding: 1.25rem 1.5rem;
    border-radius: 0 8px 8px 0;
    margin: 1.5rem 0;
    font-size: 1.05rem;
}
.purpose-box strong { color: var(--deep-blue); }

.analogy-box {
    background-color: #FFF9F0;
    border: 1px solid #FFE4B5;
    padding: 1.25rem 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.analogy-box i { font-size: 1.5rem; color: #FFA500; }
.analogy-box div { flex-grow: 1; }

/* Accordion for Line-by-Line Code */
.accordion-custom {
    margin-top: 2rem;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border-light);
}
.accordion-button {
    font-weight: 600;
    color: var(--deep-blue);
    background-color: var(--soft-gray);
    box-shadow: none !important;
}
.accordion-button:not(.collapsed) {
    color: var(--primary-accent);
    background-color: rgba(0, 102, 255, 0.05);
}
.accordion-body { padding: 0; }

/* Code Table */
.code-table { width: 100%; border-collapse: collapse; margin-bottom: 0; table-layout: fixed; }
.code-table th { background-color: var(--soft-gray); padding: 0.75rem 1.5rem; font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); border-bottom: 1px solid var(--border-light); }
.code-table th:first-child { width: 50%; }
.code-table td { padding: 0.75rem 1.5rem; border-bottom: 1px solid var(--border-light); vertical-align: middle; }
.code-table tr:hover td { background-color: #FAFAFA; }
.code-table tr:last-child td { border-bottom: none; }

.code-snippet {
    background-color: var(--code-bg);
    color: var(--code-text);
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-family: 'Fira Code', 'Courier New', monospace;
    font-size: 0.85rem;
    white-space: pre-wrap;
    word-break: break-all;
    display: block;
    line-height: 1.5;
}
.code-snippet span.keyword { color: #FF7B72; font-weight: bold; }
.code-snippet span.string { color: #A5D6FF; }
.code-snippet span.comment { color: #8B949E; font-style: italic; }
.desc-text { font-size: 0.95rem; color: var(--text-main); line-height: 1.5; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* Offcanvas mobile sidebar override */
@media (max-width: 991.98px) {
    .sidebar { position: fixed; height: 100vh; top: 0; z-index: 1045; transform: translateX(-100%); transition: transform 0.3s ease-in-out; border-right: none; box-shadow: 2px 0 10px rgba(0,0,0,0.1); width: 280px; }
    .sidebar.show { transform: translateX(0); }
    .main-content { padding: 1.5rem; }
    .doc-card-header, .doc-card-body { padding: 1.5rem; }
    .code-table th:first-child { width: auto; }
    .code-table th, .code-table td { display: block; width: 100%; border-bottom: none; }
    .code-table tr { border-bottom: 1px solid var(--border-light); display: block; }
    .code-table td:first-child { padding-bottom: 0; }
}
"""

def simple_syntax_highlight(line: str) -> str:
    """Aesthetic syntax highlighting for Python code."""
    line_esc = line.replace("<", "&lt;").replace(">", "&gt;")
    
    if line_esc.strip().startswith("#"):
        return f'<span class="comment">{line_esc}</span>'
    if '"""' in line_esc or "'''" in line_esc:
        return f'<span class="string">{line_esc}</span>'
        
    keywords = ["def ", "class ", "return ", "if ", "else:", "elif ", "for ", "in ", "while ", "import ", "from ", "async ", "await ", "try:", "except", "finally:", "True", "False", "None", "pass", "continue", "break"]
    for kw in keywords:
        if kw in line_esc:
            line_esc = line_esc.replace(kw, f'<span class="keyword">{kw}</span>')
            
    import re
    line_esc = re.sub(r'("[^"]*")', r'<span class="string">\1</span>', line_esc)
    line_esc = re.sub(r"('[^']*')", r'<span class="string">\1</span>', line_esc)
    
    return line_esc

def explain_line_simple(line: str) -> str:
    """Translate logic conceptually for clients."""
    line = line.strip()
    if not line: return "Empty space left by the developer to separate thoughts cleanly."
    if line.startswith("#"): return "A human note. The computer completely ignores this."
    if line.startswith('"""') or line.startswith("'''"): return "Comprehensive description detailing what this entire section is designed to achieve."
    
    if line.startswith("import ") or line.startswith("from "): return "Brings in powerful external tools so we don't have to reinvent the wheel."
    if line.startswith("def "): return "Defines a sequence of instructions so the system can repeatedly perform this action when asked."
    if line.startswith("class "): return "Acts as a primary blueprint. It groups related tools and data together securely."
    if line.startswith("if ") or line.startswith("elif "): return "It checks the current situation and makes a logical decision on what to do next."
    if line.startswith("else:"): return "The backup plan: if the previous conditions weren't met, it defaults to this action."
    if line.startswith("for ") or line.startswith("while "): return "Automatically loops through multiple items sequentially incredibly fast."
    if line.startswith("return "): return "Calculates the final answer and officially hands it back to the system."
    if line.startswith("try:") or line.startswith("except"): return "Safety net: Attempts an action carefully, and if an error occurs, handles it gracefully without crashing."
    if " = " in line: return "Calculates a specific value and remembers it by storing it in internal memory."
    if line.endswith("()"): return "Triggers another internal mechanism or tool."
    
    return "Executes foundational mathematical operations, processes data, or updates values."

def generate_analogies(func_name: str) -> str:
    func_l = func_name.lower()
    if "load" in func_l or "read" in func_l or "parse" in func_l:
        return "Think of this function as a librarian safely retrieving the exact document you asked for from a massive archive."
    if "save" in func_l or "write" in func_l or "register" in func_l:
        return "Think of this function as a secure vault depositing important information safely so it's never lost."
    if "detect" in func_l or "recognize" in func_l or "find" in func_l:
        return "Operates precisely like a highly trained digital security guard inspecting individuals instantly."
    if "predict" in func_l or "calculate" in func_l or "similarity" in func_l:
        return "Acts like an expert appraiser, mathematically comparing two facial signatures to determine their exact similarity."
    if "gui" in func_l or "window" in func_l or "button" in func_l:
        return "This component is the 'dashboard' of the car, allowing humans to easily drive complex machinery."
    return "This operates as a highly specialized worker on an assembly line, performing one specific task flawlessly."

def get_files():
    files = []
    for root, _, filenames in os.walk(ROOT_DIR):
        if "__pycache__" in root or ".venv" in root or ".git" in root or "embeddings" in root:
            continue
        for fn in filenames:
            if fn.endswith(".py") and fn != "generate_docs.py":
                files.append(os.path.join(root, fn))
    return sorted(files)

def build():
    files = get_files()
    structure = []
    
    for fw in files:
        rel_path = os.path.relpath(fw, ROOT_DIR)
        with open(fw, "r", encoding="utf-8") as f:
            code_lines = f.readlines()
            
        try:
            tree = ast.parse("".join(code_lines))
        except:
            continue
            
        funcs = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                funcs.append({
                    "name": node.name,
                    "type": "Function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "Class",
                    "start": node.lineno,
                    "end": node.end_lineno,
                    "doc": ast.get_docstring(node) or "Executes proprietary backend logic systematically isolated from the interface.",
                })
        
        if funcs:
            structure.append({
                "file": rel_path,
                "lines": code_lines,
                "items": funcs
            })

    # Sidebar Generation
    nav_html = ""
    content_html = ""
    
    # Overview Content
    content_html += '''
    <div id="section-overview" class="doc-card">
        <div class="doc-card-header">
            <h2><i class="bi bi-house-door-fill text-primary me-2"></i> Project Overview</h2>
        </div>
        <div class="doc-card-body">
            <h4 class="mb-3 text-dark font-weight-bold">Solving Identify Verification Sensibly</h4>
            <p class="lead">This software implements state-of-the-art artificial intelligence to provide secure "Few-Shot" facial recognition. Most systems require thousands of images per person. Our system securely maps geometric facial features using just a handful of images, dramatically boosting efficiency while maintaining strict privacy standards.</p>
            
            <div class="row g-4 mt-2">
                <div class="col-md-4">
                    <div class="p-4 bg-light rounded shadow-sm border h-100">
                        <i class="bi bi-camera-fill text-primary" style="font-size: 2rem;"></i>
                        <h5 class="mt-3 font-weight-bold">1. Collection</h5>
                        <p class="text-muted small">Securely captures physical imagery mathematically without storing raw video persistently.</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="p-4 bg-light rounded shadow-sm border h-100">
                        <i class="bi bi-cpu-fill text-primary" style="font-size: 2rem;"></i>
                        <h5 class="mt-3 font-weight-bold">2. Deconstruction</h5>
                        <p class="text-muted small">Transforms human faces into deep structural numerical signatures using neural networks.</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="p-4 bg-light rounded shadow-sm border h-100">
                        <i class="bi bi-shield-check-fill text-success" style="font-size: 2rem;"></i>
                        <h5 class="mt-3 font-weight-bold">3. Verification</h5>
                        <p class="text-muted small">Performs rapid statistical similarity comparisons to flawlessly identify subjects in real-time.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div id="section-structure" class="doc-card">
        <div class="doc-card-header">
            <h2><i class="bi bi-folder-fill text-warning me-2"></i> Architecture Layout</h2>
        </div>
        <div class="doc-card-body">
            <p class="mb-4">To ensure seamless scalability and complete transparency, the software is divided into discrete, purposeful specialized files. Each file operates like an independent corporate department.</p>
            <div class="row g-3">
    '''
    
    for f in structure:
        slug_file = f"file_{f['file'].replace('.py', '').replace(os.sep, '_')}"
        content_html += f'''
                <div class="col-md-6">
                    <div class="p-3 border rounded h-100 d-flex flex-column" style="background-color: var(--soft-gray);">
                        <div class="d-flex align-items-center mb-2">
                            <i class="bi bi-filetype-py fs-4 text-primary me-2"></i>
                            <h6 class="mb-0 fw-bold">{f["file"]}</h6>
                        </div>
                        <p class="text-muted small mb-0 mt-auto">Manages {len(f["items"])} critical architectural mechanisms.</p>
                    </div>
                </div>
        '''
        
    content_html += '''
            </div>
        </div>
    </div>
    '''

    nav_html += '<a href="#section-overview" class="nav-link active"><i class="bi bi-info-circle me-2"></i> Overview</a>'
    nav_html += '<a href="#section-structure" class="nav-link"><i class="bi bi-diagram-3 me-2"></i> Architecture</a>'

    accordion_id_counter = 0

    for file_info in structure:
        nav_html += f'<div class="nav-group">{file_info["file"]}</div>'
        
        for item in file_info["items"]:
            slug = f"doc_{file_info['file'].replace('.py', '').replace(os.sep, '_')}_{item['name']}"
            nav_html += f'<a href="#{slug}" class="nav-link"><i class="bi bi-dot"></i> {item["name"]}</a>'
            
            start = item["start"] - 1
            end = item["end"]
            chunk = file_info["lines"][start:end]

            analogy = generate_analogies(item["name"])

            args_str = "Utilizes internally pre-configured properties dynamically."
            if "def " in chunk[0] and "(" in chunk[0] and ")" in "".join(chunk[:3]):
                args_str = "Actively requires variables (ingredients) injected from external processes to compute outputs."

            accordion_id_counter += 1
            accordion_id = f"flush-collapse-{accordion_id_counter}"

            content_html += f'''
            <div id="{slug}" class="doc-card doc-function">
                <div class="doc-card-header d-flex justify-content-between align-items-center flex-wrap gap-2">
                    <h2><i class="bi bi-gear-fill me-2 text-secondary"></i> {item['name']} <span class="badge bg-primary rounded-pill ms-3">{item['type']}</span></h2>
                </div>
                
                <div class="doc-card-body">
                    <div class="info-block">
                        <div class="info-label"><i class="bi bi-geo-alt-fill me-1"></i> File Location:</div>
                        <div class="info-desc font-monospace text-primary">{file_info['file']}</div>
                    </div>
                    <div class="info-block">
                        <div class="info-label"><i class="bi bi-arrow-down-up me-1"></i> Inputs & Outputs:</div>
                        <div class="info-desc">{args_str}</div>
                    </div>

                    <div class="purpose-box">
                        <strong>Developer Instruction:</strong><br>
                        <span style="white-space: pre-wrap;">{item['doc']}</span>
                    </div>

                    <div class="analogy-box shadow-sm">
                        <i class="bi bi-lightbulb-fill"></i>
                        <div>
                            <strong>Client Analogy</strong><br>
                            <span class="text-secondary">{analogy}</span>
                        </div>
                    </div>

                    <div class="accordion accordion-flush accordion-custom" id="accordionFlushExample_{accordion_id_counter}">
                        <div class="accordion-item">
                            <h2 class="accordion-header">
                                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#{accordion_id}" aria-expanded="false" aria-controls="{accordion_id}">
                                    <i class="bi bi-code-square me-2"></i> View Line-by-Line Technical Translation
                                </button>
                            </h2>
                            <div id="{accordion_id}" class="accordion-collapse collapse" data-bs-parent="#accordionFlushExample_{accordion_id_counter}">
                                <div class="accordion-body">
                                    <table class="code-table">
                                        <thead>
                                            <tr>
                                                <th>Raw Source Code</th>
                                                <th>Plain English Logic</th>
                                            </tr>
                                        </thead>
                                        <tbody>
            '''
            
            for row in chunk:
                html_code = simple_syntax_highlight(row)
                meaning = explain_line_simple(row)
                content_html += f'''
                                            <tr>
                                                <td><div class="code-snippet">{html_code.rstrip()}</div></td>
                                                <td><div class="desc-text">{meaning}</div></td>
                                            </tr>
                '''
                
            content_html += '''
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            '''

    # Final HTML assembly
    html_out = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Professional Documentation System</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code&display=swap" rel="stylesheet">
    <style>{CSS_CONTENT}</style>
</head>
<body data-bs-spy="scroll" data-bs-target="#sidebarMenu" data-bs-offset="100">

    <!-- Top Navbar -->
    <nav class="navbar navbar-expand-lg navbar-custom fixed-top">
        <div class="container-fluid px-0">
            <button class="navbar-toggler border-0 me-2" type="button" data-bs-toggle="offcanvas" data-bs-target="#mobileSidebar" aria-controls="mobileSidebar">
                <i class="bi bi-list fs-2 text-dark"></i>
            </button>
            <a class="navbar-brand d-flex align-items-center" href="#">
                <i class="bi bi-fingerprint"></i> FewShotFace <span class="ms-2 fw-normal fs-6 text-muted d-none d-sm-inline border-start ps-2">Enterprise Documentation</span>
            </a>
        </div>
    </nav>

    <div class="container-fluid">
        <div class="row">
            <!-- Desktop Sidebar -->
            <nav id="sidebarMenu" class="col-md-4 col-lg-3 col-xl-2 d-none d-lg-block sidebar">
                <div class="search-container">
                    <div class="input-group">
                        <span class="input-group-text bg-light border-end-0 text-muted"><i class="bi bi-search"></i></span>
                        <input type="text" class="form-control bg-light border-start-0 ps-0" id="desktopSearch" placeholder="Search functions..." onkeyup="filterSidebar('desktopSearch', '#sidebarMenu')">
                    </div>
                </div>
                <div class="nav flex-column sidebar-nav-container">
                    {nav_html}
                </div>
            </nav>

            <!-- Mobile Offcanvas Sidebar -->
            <div class="offcanvas offcanvas-start border-0 shadow" tabindex="-1" id="mobileSidebar" aria-labelledby="mobileSidebarLabel">
                <div class="offcanvas-header border-bottom">
                    <h5 class="offcanvas-title fw-bold text-primary" id="mobileSidebarLabel"><i class="bi bi-fingerprint"></i> Documentation</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="offcanvas" aria-label="Close"></button>
                </div>
                <div class="offcanvas-body sidebar w-100 position-relative h-100 border-0 shadow-none mt-0 pt-3">
                    <div class="search-container">
                        <div class="input-group">
                            <span class="input-group-text bg-light border-end-0 text-muted"><i class="bi bi-search"></i></span>
                            <input type="text" class="form-control bg-light border-start-0 ps-0" id="mobileSearch" placeholder="Search functions..." onkeyup="filterSidebar('mobileSearch', '#mobileSidebar')">
                        </div>
                    </div>
                    <div class="nav flex-column sidebar-nav-container">
                        {nav_html}
                    </div>
                </div>
            </div>

            <!-- Main Content Area -->
            <main class="col-lg-9 col-xl-10 ms-sm-auto px-0">
                <div class="main-content">
                    <div class="section-header">
                        <h1>Technical Architecture & Research</h1>
                        <p class="lead">A completely transparent, logical translation of system components presented cleanly for high-level technical assessment.</p>
                    </div>
                    
                    {content_html}
                    
                    <footer class="mt-5 mb-3 text-center text-muted small">
                        Auto-generated comprehensive documentation interface inside securely isolated logic infrastructure.
                    </footer>
                </div>
            </main>
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.css"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

    <!-- Custom Interactivity -->
    <script>
        function filterSidebar(inputId, sidebarId) {{
            let filter = document.getElementById(inputId).value.toLowerCase();
            let container = document.querySelector(sidebarId + ' .sidebar-nav-container');
            let links = container.querySelectorAll('.nav-link:not([href="#section-overview"]):not([href="#section-structure"])');
            let groups = container.querySelectorAll('.nav-group');
            
            if (filter) {{
                groups.forEach(g => g.style.display = 'none');
                links.forEach(l => {{
                    if (l.innerText.toLowerCase().includes(filter)) {{
                        l.style.display = 'block';
                        // Keep previous group visible
                        let prev = l.previousElementSibling;
                        while(prev) {{
                            if(prev.classList.contains('nav-group')) {{
                                prev.style.display = 'block';
                                break;
                            }}
                            prev = prev.previousElementSibling;
                        }}
                    }} else {{
                        l.style.display = 'none';
                    }}
                }});
            }} else {{
                groups.forEach(g => g.style.display = 'block');
                links.forEach(l => l.style.display = 'block');
            }}
        }}

        // Smooth scroll and mobile sidebar dismissal
        document.querySelectorAll('.nav-link').forEach(link => {{
            link.addEventListener('click', function(e) {{
                // If it's a mobile offcanvas link, close the offcanvas
                const offcanvasEl = document.getElementById('mobileSidebar');
                if (offcanvasEl && offcanvasEl.classList.contains('show')) {{
                    const bsOffcanvas = bootstrap.Offcanvas.getInstance(offcanvasEl) || new bootstrap.Offcanvas(offcanvasEl);
                    bsOffcanvas.hide();
                }}
            }});
        }});
    </script>
</body>
</html>'''

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_out)

if __name__ == "__main__":
    build()
    print(f"Bootstrap Presentation Documentation completely finalized at:\\n{OUTPUT_FILE}")
