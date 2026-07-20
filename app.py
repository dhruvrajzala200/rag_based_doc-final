import os
import json
import time
import uuid
import tempfile
import hashlib
import pandas as pd
import speech_recognition as sr
import streamlit as st
from rag_engine import RAGEngine

# Page Configuration
st.set_page_config(page_title="Document Intelligence System", page_icon="📄", layout="wide")

# Cache RAG Engine instance to avoid re-initialization on every rerun
@st.cache_resource
def load_rag_engine():
    return RAGEngine()

rag_engine = load_rag_engine()

# File Storage Paths
USER_DATA_PATH = "vectorstore/user_data.json"
USER_NOTES_PATH = "vectorstore/user_notes.json"
SHARED_CHAT_PATH = "vectorstore/shared_chat.json"

def read_json(filepath, default_value):
    """Utility function to read JSON data safely."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return default_value
    return default_value

def save_json(filepath, data):
    """Utility function to write JSON data safely."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except Exception as error:
        print(f"Error saving data: {error}")

# Initialize Session State Variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_query" not in st.session_state:
    st.session_state.active_query = ""
if "run_query" not in st.session_state:
    st.session_state.run_query = False
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🏠 Assistant Room"
if "processed_audio_hashes" not in st.session_state:
    st.session_state.processed_audio_hashes = set()

# Load persisted user databases
user_data = read_json(USER_DATA_PATH, {"favorites": [], "pinned": []})
user_notes = read_json(USER_NOTES_PATH, {})

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("<h2 style='color:#58a6ff;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    ollama_endpoint = st.text_input("Ollama Endpoint URL", value="http://localhost:11434")
    llm_model = st.selectbox("LLM Model", ["llama3.2", "llama3", "mistral", "gemma2"], index=0)
    embed_model = st.selectbox("Embedding Model", ["nomic-embed-text", "all-minilm"], index=0)

    st.markdown("### 🛠️ Retrieval Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=5, step=1)

    # Configure backend engine
    try:
        rag_engine.configure(
            llm_model=llm_model,
            embed_model=embed_model,
            endpoint=ollama_endpoint,
            temperature=temperature
        )
    except Exception as error:
        st.error(f"Engine setup error: {error}")

    # Fetch document list from active index
    available_docs = rag_engine.get_documents_list()
    doc_filenames = [doc["filename"] for doc in available_docs]

    st.markdown("### 📂 Document Scope")
    if doc_filenames:
        search_scope = st.radio("Scope", ["All Indexed Documents", "Selected Document(s) Only"])
        if search_scope == "Selected Document(s) Only":
            selected_query_docs = st.multiselect("Select Documents", doc_filenames, default=doc_filenames)
        else:
            selected_query_docs = doc_filenames
    else:
        st.info("No documents uploaded yet.")
        selected_query_docs = []

    # Starred Favorites Drawer
    st.markdown("### ⭐ Saved Conversations")
    favorites_list = user_data.get("favorites", [])
    if favorites_list:
        fav_options = [fav["name"] for fav in favorites_list]
        selected_fav = st.selectbox("Select Saved Chat", ["-- Choose --"] + fav_options)
        if selected_fav != "-- Choose --":
            chosen_fav = next(f for f in favorites_list if f["name"] == selected_fav)
            if st.button("Load Conversation"):
                st.session_state.messages = chosen_fav["messages"]
                st.rerun()
    else:
        st.caption("No saved conversations.")

    # Pinned Answers Drawer
    st.markdown("### 📌 Pinned Answers")
    pinned_list = user_data.get("pinned", [])
    if pinned_list:
        for idx, pin in enumerate(pinned_list):
            with st.expander(f"📌 {pin['question'][:30]}..."):
                st.write(pin["answer"])
                if st.button("Remove Pin", key=f"unpin_{idx}"):
                    pinned_list.pop(idx)
                    user_data["pinned"] = pinned_list
                    save_json(USER_DATA_PATH, user_data)
                    st.rerun()
    else:
        st.caption("No pinned answers.")

    # Export Options
    st.markdown("### 📥 Export Chat")
    if st.session_state.messages:
        plain_text = ""
        markdown_text = f"# RAG Chat Export ({llm_model})\n\n"
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            plain_text += f"{role}: {msg['content']}\n\n"
            markdown_text += f"**{role}**: {msg['content']}\n\n"

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.download_button("TXT", plain_text, "chat_export.txt", "text/plain")
        with col_e2:
            st.download_button("MD", markdown_text, "chat_export.md", "text/markdown")

# Apply Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .main-header {
        background: -webkit-linear-gradient(45deg, #58a6ff, #bc3cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.1rem;
    }
    .topic-badge {
        background-color: #21262d;
        border: 1px solid #30363d;
        color: #58a6ff;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.75rem;
        margin-right: 6px;
        display: inline-block;
        margin-bottom: 4px;
    }
    .citation-box {
        background-color: #161b22;
        border-left: 3px solid #58a6ff;
        padding: 10px;
        margin-top: 6px;
        font-size: 0.85rem;
        color: #8b949e;
        border-radius: 0 6px 6px 0;
    }
    .metric-card {
        background-color: #1f242c;
        border-radius: 6px;
        padding: 8px;
        text-align: center;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown("<h1 class='main-header'>📄 Document Intelligence System</h1>", unsafe_allow_html=True)
st.caption(f"Engine: Ollama (`{llm_model}`) | Vector Model: (`{embed_model}`)")

# Check if Ollama endpoint is reachable
is_ollama_online = rag_engine.check_connection()
if not is_ollama_online:
    st.warning(
        f"⚠️ **Cannot connect to Ollama** at `{ollama_endpoint}`.\n\n"
        "If you are running on **Streamlit Cloud**, `http://localhost:11434` cannot connect to your laptop directly. "
        "Please expose your local Ollama port via Ngrok (`ngrok http 11434`) and enter your public URL (e.g. `https://xxxx.ngrok-free.app`) in the sidebar."
    )

# Horizontal Main Navigation Bar
workspace_tabs = [
    "🏠 Assistant Room",
    "📁 Document Manager",
    "📄 Side-by-Side Comparison",
    "👥 Collaborative Chat"
]

if st.session_state.active_tab not in workspace_tabs:
    st.session_state.active_tab = "🏠 Assistant Room"

nav_columns = st.columns(4)
for index, tab_name in enumerate(workspace_tabs):
    button_style = "primary" if st.session_state.active_tab == tab_name else "secondary"
    if nav_columns[index].button(tab_name, key=f"nav_button_{index}", type=button_style, use_container_width=True):
        st.session_state.active_tab = tab_name
        st.rerun()

st.markdown("---")

# Helper to render interactive Vis.js Knowledge Graph
def render_knowledge_graph(kg_data):
    nodes = kg_data.get("nodes", [])
    edges = kg_data.get("edges", [])
    if not nodes:
        st.info("No knowledge graph entities found.")
        return
        
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            #vis_container {{
                width: 100%;
                height: 380px;
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div id="vis_container"></div>
        <script>
            var nodes = new vis.DataSet({nodes_json});
            var edges = new vis.DataSet({edges_json});
            var container = document.getElementById('vis_container');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                nodes: {{
                    shape: 'dot',
                    size: 16,
                    font: {{ color: '#c9d1d9', size: 12 }},
                    borderWidth: 2,
                    color: {{ background: '#21262d', border: '#58a6ff', highlight: {{ background: '#58a6ff', border: '#bc3cff' }} }}
                }},
                edges: {{
                    color: '#30363d',
                    font: {{ color: '#8b949e', size: 10 }},
                    arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }}
                }},
                physics: {{ barnesHut: {{ gravitationalConstant: -2000, centralGravity: 0.3, springLength: 95 }} }}
            }};
            new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=400)


# --- WORKSPACE 1: Assistant Room ---
if st.session_state.active_tab == "🏠 Assistant Room":
    st.markdown("### 💬 Chat Assistant")

    # Audio Recording Widget
    recorded_audio = st.audio_input("🎤 Voice Input (Record and Transcribe)")
    if recorded_audio is not None:
        audio_payload = recorded_audio.getvalue()
        audio_hash = hashlib.md5(audio_payload).hexdigest()
        
        # Process audio payload only once using MD5 hashing
        if audio_hash not in st.session_state.processed_audio_hashes:
            st.session_state.processed_audio_hashes.add(audio_hash)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_payload)
                temp_filepath = temp_file.name
            
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_filepath) as source:
                    audio_content = recognizer.record(source)
                transcribed_text = recognizer.recognize_google(audio_content)
                
                if transcribed_text.strip():
                    st.session_state.active_query = transcribed_text
                    st.session_state.run_query = True
                    st.toast(f"Transcribed: {transcribed_text}")
            except Exception as err:
                st.error(f"Speech recognition error: {err}")
            finally:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)

    # Render Chat Messages History
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Controls for Assistant Responses
            if message["role"] == "assistant":
                col_a1, col_a2 = st.columns([1, 5])
                with col_a1:
                    # Client-side Text-to-Speech button
                    safe_text = message['content'].replace("'", "\\'").replace("\n", " ").replace('"', '\\"')
                    tts_button_html = f"""
                    <button id="speak_btn_{idx}" style="background:#21262d; color:#58a6ff; border:1px solid #30363d; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:11px;">🔊 Listen</button>
                    <script>
                        document.getElementById('speak_btn_{idx}').addEventListener('click', function() {{
                            window.speechSynthesis.cancel();
                            var speech = new SpeechSynthesisUtterance("{safe_text}");
                            window.speechSynthesis.speak(speech);
                        }});
                    </script>
                    """
                    st.components.v1.html(tts_button_html, height=35)

                with col_a2:
                    if st.button("📌 Pin Answer", key=f"pin_button_{idx}"):
                        question_text = st.session_state.messages[idx - 1]["content"] if idx > 0 else "Query"
                        new_pin_item = {
                            "question": question_text,
                            "answer": message["content"],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        if new_pin_item not in user_data["pinned"]:
                            user_data["pinned"].append(new_pin_item)
                            save_json(USER_DATA_PATH, user_data)
                            st.toast("Answer pinned!")
                            st.rerun()

            # Source Citations
            if message["role"] == "assistant" and message.get("citations"):
                with st.expander("📖 View Citations"):
                    for c_idx, cite in enumerate(message["citations"]):
                        src_file = cite.get("source")
                        src_page = cite.get("page") + 1
                        excerpt = cite.get("content")
                        st.markdown(f"""
                        <div class='citation-box'>
                            <strong>[{c_idx+1}] Source: {src_file} (Page {src_page})</strong><br>
                            <em>"{excerpt}"</em>
                        </div>
                        """, unsafe_allow_html=True)

            # Response Analytics
            if message["role"] == "assistant" and message.get("analytics"):
                analytics = message["analytics"]
                ret_sec = round(analytics.get("retrieval_time", 0.0), 3)
                gen_sec = round(analytics.get("generation_time", 0.0), 3)
                tot_sec = round(ret_sec + gen_sec, 3)

                st.markdown(f"""
                <div style='display: flex; gap: 10px; margin-top: 8px;'>
                    <div class='metric-card' style='flex: 1;'>⏱️ Retrieval: <strong>{ret_sec}s</strong></div>
                    <div class='metric-card' style='flex: 1;'>⚡ Generation: <strong>{gen_sec}s</strong></div>
                    <div class='metric-card' style='flex: 1;'>📊 Total: <strong>{tot_sec}s</strong></div>
                </div>
                """, unsafe_allow_html=True)

    # Input Box for User Queries
    input_text = st.chat_input("Ask a question about your uploaded document(s)...")

    active_user_query = st.session_state.active_query
    if input_text:
        active_user_query = input_text
        st.session_state.run_query = True

    # Process and Stream RAG Response
    if st.session_state.run_query and active_user_query:
        st.session_state.run_query = False
        st.session_state.active_query = ""

        # Display User Message
        st.session_state.messages.append({"role": "user", "content": active_user_query})
        with st.chat_message("user"):
            st.markdown(active_user_query)

        # Stream Assistant Response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            token_stream = rag_engine.query_stream(
                question=active_user_query,
                selected_files=selected_query_docs,
                k=top_k
            )

            full_answer = ""
            for chunk in token_stream:
                full_answer += chunk
                placeholder.markdown(full_answer)

            # Collect citations and timing metrics
            last_metadata = getattr(rag_engine, "last_query_metadata", {})
            citation_data = []
            if "retrieved_docs" in last_metadata:
                for doc in last_metadata["retrieved_docs"]:
                    citation_data.append({
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page", 0),
                        "content": doc.page_content
                    })

            analytics_data = {
                "retrieval_time": last_metadata.get("retrieval_time", 0.0),
                "generation_time": last_metadata.get("generation_time", 0.0)
            }

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "citations": citation_data,
                "analytics": analytics_data
            })
            st.rerun()

    # Save Session and Clear Actions
    if st.session_state.messages:
        st.markdown("---")
        col_s1, col_s2 = st.columns([4, 1])
        with col_s1:
            save_name = st.text_input("Name this conversation to save:", placeholder="e.g. Q3 Compliance Review")
        with col_s2:
            if st.button("⭐ Save Chat"):
                if save_name.strip():
                    new_fav = {
                        "name": save_name.strip(),
                        "messages": st.session_state.messages,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    user_data["favorites"].append(new_fav)
                    save_json(USER_DATA_PATH, user_data)
                    st.toast("Saved to Favorites!")
                    st.rerun()
                else:
                    st.warning("Please enter a name.")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


# --- WORKSPACE 2: Document Manager ---
elif st.session_state.active_tab == "📁 Document Manager":
    st.markdown("### 📚 Document Manager")

    # PDF Uploader Section
    pdf_uploads = st.file_uploader("Upload PDF Documents (Automatic versioning supported)", type=["pdf"], accept_multiple_files=True)
    if pdf_uploads:
        if st.button("🚀 Process Uploads"):
            if not is_ollama_online:
                st.error("❌ Ollama server is offline or unreachable. Please check your Ollama Endpoint URL in the sidebar before processing uploads.")
            else:
                has_error = False
                with st.spinner("Processing document semantics and creating index..."):
                    for pdf in pdf_uploads:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                            temp.write(pdf.getvalue())
                            temp_path = temp.name
                        try:
                            chunk_count = rag_engine.process_document(temp_path, pdf.name)
                            st.success(f"Indexed **{pdf.name}** into {chunk_count} chunks!")
                        except Exception as err:
                            has_error = True
                            st.error(f"❌ Error processing **{pdf.name}**: {err}\n\nMake sure your Ollama URL is accessible.")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                if not has_error:
                    st.rerun()

    # Render Document Cards and Dynamic Features
    if available_docs:
        for doc in available_docs:
            doc_name = doc["filename"]
            st.markdown(f"#### 📁 {doc_name}")

            col_m1, col_m2, col_m3 = st.columns([2, 2, 1])
            with col_m1:
                selected_version = st.selectbox(
                    f"Version ({doc_name})",
                    doc["versions_list"],
                    index=doc["versions_list"].index(doc["active_version"]),
                    key=f"version_select_{doc_name}"
                )
                if selected_version != doc["active_version"]:
                    if rag_engine.set_active_version(doc_name, selected_version):
                        st.toast(f"Active version set to {selected_version}")
                        st.rerun()

            with col_m2:
                action_choice = st.selectbox("Action", ["-- Choose Action --", "Delete Selected Version", "Delete Entire Document"], key=f"action_select_{doc_name}")
                if action_choice != "-- Choose Action --":
                    if action_choice == "Delete Selected Version":
                        if rag_engine.delete_document(doc_name, selected_version):
                            st.toast(f"Deleted version {selected_version}")
                            st.rerun()
                    elif action_choice == "Delete Entire Document":
                        if rag_engine.delete_document(doc_name):
                            st.toast(f"Deleted {doc_name}")
                            st.rerun()

            with col_m3:
                file_kb = round(doc['size_bytes'] / 1024, 2)
                st.markdown(f"**Size**: {file_kb} KB<br>**Chunks**: {doc['chunks_count']}", unsafe_allow_html=True)

            # Dynamic Suggested Questions (Top-Level Button Placement)
            suggestions = rag_engine.generate_suggested_questions([doc_name])
            if suggestions:
                st.markdown("💡 **Dynamic Suggestions** *(Click any question to ask in Assistant Room)*:")
                for s_idx, question in enumerate(suggestions):
                    if st.button(f"❓ {question}", key=f"suggestion_{doc_name}_{s_idx}_{selected_version}"):
                        st.session_state.active_tab = "🏠 Assistant Room"
                        st.session_state.active_query = question
                        st.session_state.run_query = True
                        st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)

            # Document Details Expander
            with st.expander(f"🔍 View Details & Features for {doc_name} ({selected_version})"):
                doc_tabs = st.tabs(["📝 Summary & Topics", "📷 Extracted Images", "🧠 Knowledge Graph", "📊 Metrics Chart", "✍️ Notes & Bookmarks"])

                # Summary Tab
                with doc_tabs[0]:
                    st.write(f"**Executive Summary**: *{doc['summary']}*")
                    st.markdown("**Key Topics**:")
                    badges_html = " ".join([f"<span class='topic-badge'>{k}</span>" for k in doc['keywords']])
                    st.markdown(badges_html, unsafe_allow_html=True)

                # Images Gallery Tab
                with doc_tabs[1]:
                    extracted_imgs = doc.get("images", [])
                    if extracted_imgs:
                        st.write(f"Extracted {len(extracted_imgs)} image(s):")
                        image_cols = st.columns(3)
                        for img_idx, img_path in enumerate(extracted_imgs):
                            col_pos = img_idx % 3
                            if os.path.exists(img_path):
                                image_cols[col_pos].image(img_path, caption=f"Page Image {img_idx + 1}")
                    else:
                        st.info("No embedded images extracted.")

                # Knowledge Graph Tab
                with doc_tabs[2]:
                    graph_data = doc.get("knowledge_graph", {"nodes": [], "edges": []})
                    if graph_data.get("nodes"):
                        st.markdown("**Interactive Knowledge Graph**:")
                        render_knowledge_graph(graph_data)
                    else:
                        st.info("No graph data extracted.")

                # Metrics Plotting Tab
                with doc_tabs[3]:
                    chart_items = doc.get("data_charts", [])
                    if chart_items:
                        df_charts = pd.DataFrame(chart_items)
                        if "label" in df_charts.columns and "value" in df_charts.columns:
                            df_charts["value"] = pd.to_numeric(df_charts["value"], errors="coerce")
                            df_clean = df_charts.dropna(subset=["value"])
                            if not df_clean.empty:
                                st.bar_chart(df_clean.set_index("label"))
                            else:
                                st.info("No valid numerical data.")
                    else:
                        st.info("No numerical metrics found.")

                # Bookmarks and Notes Tab
                with doc_tabs[4]:
                    st.write("📝 **Page Notes & Bookmarks**:")
                    saved_notes = user_notes.get(doc_name, {}).get(selected_version, [])
                    
                    if saved_notes:
                        for n_idx, item in enumerate(saved_notes):
                            col_n1, col_n2 = st.columns([5, 1])
                            with col_n1:
                                st.markdown(f"🔖 **Page {item['page']}**: *{item['note']}* ({item['timestamp']})")
                            with col_n2:
                                if st.button("Delete", key=f"del_note_{doc_name}_{selected_version}_{n_idx}"):
                                    saved_notes.pop(n_idx)
                                    user_notes[doc_name][selected_version] = saved_notes
                                    save_json(USER_NOTES_PATH, user_notes)
                                    st.rerun()
                    else:
                        st.caption("No notes added yet.")

                    # Form to add a new note
                    st.markdown("Add a new note:")
                    col_p, col_t = st.columns([1, 4])
                    with col_p:
                        page_num = st.number_input("Page", min_value=1, step=1, key=f"page_in_{doc_name}_{selected_version}")
                    with col_t:
                        note_text = st.text_input("Note Content", key=f"text_in_{doc_name}_{selected_version}")

                    if st.button("Add Note", key=f"save_note_{doc_name}_{selected_version}"):
                        if note_text.strip():
                            if doc_name not in user_notes:
                                user_notes[doc_name] = {}
                            if selected_version not in user_notes[doc_name]:
                                user_notes[doc_name][selected_version] = []

                            user_notes[doc_name][selected_version].append({
                                "page": page_num,
                                "note": note_text.strip(),
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            save_json(USER_NOTES_PATH, user_notes)
                            st.toast("Note saved!")
                            st.rerun()

            st.markdown("---")
    else:
        st.info("No documents indexed yet. Upload a PDF above.")


# --- WORKSPACE 3: Side-by-Side Comparison ---
elif st.session_state.active_tab == "📄 Side-by-Side Comparison":
    st.markdown("### 🧾 Comparative Analysis")

    if len(doc_filenames) >= 2:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            file_a = st.selectbox("Document A", doc_filenames, key="compare_doc_a")
            versions_a = next(d["versions_list"] for d in available_docs if d["filename"] == file_a)
            ver_a = st.selectbox("Version A", versions_a, key="compare_ver_a")
        with col_c2:
            file_b = st.selectbox("Document B", doc_filenames, key="compare_doc_b")
            versions_b = next(d["versions_list"] for d in available_docs if d["filename"] == file_b)
            ver_b = st.selectbox("Version B", versions_b, key="compare_ver_b")

        if st.button("📊 Compare Documents"):
            with st.spinner("Analyzing differences..."):
                stream = rag_engine.compare_documents(file_a, ver_a, file_b, ver_b)
                comparison_box = st.empty()
                accumulated_text = ""
                for chunk in stream:
                    accumulated_text += chunk
                    comparison_box.markdown(accumulated_text)
    else:
        st.info("Upload at least 2 PDF documents to run a side-by-side comparison.")


# --- WORKSPACE 4: Collaborative Chat ---
elif st.session_state.active_tab == "👥 Collaborative Chat":
    st.markdown("### 📡 Collaborative Chat Room")
    st.caption("Shared channel across all local browser sessions.")

    shared_chat_messages = read_json(SHARED_CHAT_PATH, [])

    # Display shared feed
    chat_container = st.container(height=300)
    with chat_container:
        if shared_chat_messages:
            for s_msg in shared_chat_messages:
                st.markdown(f"👤 **{s_msg['user']}** *[{s_msg['timestamp']}]*: {s_msg['content']}")
        else:
            st.info("Room is currently empty.")

    # Post message form
    with st.form("shared_room_form", clear_on_submit=True):
        col_w1, col_w2 = st.columns([5, 1])
        with col_w1:
            room_input = st.text_input("Message", placeholder="Post to shared room...")
        with col_w2:
            send_room = st.form_submit_button("Send")

        if send_room and room_input.strip():
            new_room_msg = {
                "user": f"User-{st.session_state.session_id[:6]}",
                "content": room_input.strip(),
                "timestamp": time.strftime("%H:%M:%S")
            }
            shared_chat_messages.append(new_room_msg)
            save_json(SHARED_CHAT_PATH, shared_chat_messages)
            st.rerun()

    col_r1, col_r2 = st.columns([5, 1])
    with col_r2:
        if st.button("🔄 Sync Feed"):
            st.rerun()
    with col_r1:
        if st.button("🗑️ Reset Room"):
            save_json(SHARED_CHAT_PATH, [])
            st.rerun()
