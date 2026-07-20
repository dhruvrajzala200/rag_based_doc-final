import os
import json
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader

class RAGEngine:
    def __init__(self):
        self.llm_model = "llama3.2"
        self.embed_model = "nomic-embed-text"
        self.endpoint = "http://localhost:11434"
        self.temperature = 0.3
        
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.index_dir = None
        self.metadata = {}
        self.metadata_path = os.path.join("vectorstore", "metadata.json")
        self.last_query_metadata = {}

    def configure(self, llm_model, embed_model, endpoint=None, temperature=0.3):
        """Initializes the Ollama models and loads the persistent vector index."""
        self.llm_model = llm_model
        self.embed_model = embed_model
        
        # Normalize endpoint URL
        if endpoint:
            ep = endpoint.strip().rstrip("/")
            if not ep.startswith("http://") and not ep.startswith("https://"):
                ep = "https://" + ep
            self.endpoint = ep
        else:
            self.endpoint = "http://localhost:11434"
            
        self.temperature = temperature
        
        # Headers to bypass Ngrok browser warning page
        request_headers = {
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "Mozilla/5.0"
        }
        
        self.embeddings = OllamaEmbeddings(
            model=self.embed_model,
            base_url=self.endpoint,
            sync_client_kwargs={"headers": request_headers}
        )

        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.temperature,
            base_url=self.endpoint,
            sync_client_kwargs={"headers": request_headers}
        )

        # Vector store folder path based on the embedding model
        safe_model_name = self.embed_model.replace("/", "_")
        self.index_dir = os.path.join("vectorstore", "indices", f"Ollama_{safe_model_name}")
        
        if os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            self.vector_store = FAISS.load_local(
                self.index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None

        # Load metadata database
        os.makedirs("vectorstore", exist_ok=True)
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {}

    def check_connection(self):
        """Checks if the configured Ollama server endpoint is reachable."""
        try:
            import urllib.request
            url = f"{self.endpoint}/api/tags"
            req = urllib.request.Request(
                url, 
                headers={
                    "ngrok-skip-browser-warning": "true",
                    "User-Agent": "Mozilla/5.0"
                }
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception as err:
            print(f"Connection check failed for {self.endpoint}: {err}")
            return False

    def get_documents_list(self):
        """Returns details for all active indexed documents."""
        document_list = []
        for filename, data in self.metadata.items():
            active_ver = data.get("active_version", "v1")
            versions = data.get("versions", {})
            if active_ver in versions:
                v_data = versions[active_ver]
                document_list.append({
                    "filename": filename,
                    "active_version": active_ver,
                    "versions_list": list(versions.keys()),
                    "chunks_count": v_data.get("chunks_count", 0),
                    "size_bytes": v_data.get("size_bytes", 0),
                    "uploaded_at": v_data.get("uploaded_at", ""),
                    "summary": v_data.get("summary", ""),
                    "keywords": v_data.get("keywords", []),
                    "images": v_data.get("images", []),
                    "knowledge_graph": v_data.get("knowledge_graph", {"nodes": [], "edges": []}),
                    "data_charts": v_data.get("data_charts", [])
                })
        return document_list

    def delete_document(self, filename, version=None):
        """Deletes a single version or an entire document from index and disk."""
        if filename not in self.metadata:
            return False

        versions_to_delete = [version] if version else list(self.metadata[filename]["versions"].keys())

        # Remove matching document chunks from vector store
        if self.vector_store is not None:
            ids_to_delete = []
            for doc_id, doc in self.vector_store.docstore._dict.items():
                match_file = doc.metadata.get("source") == filename
                match_version = doc.metadata.get("version") in versions_to_delete
                if match_file and match_version:
                    ids_to_delete.append(doc_id)
            
            if ids_to_delete:
                self.vector_store.delete(ids_to_delete)
                if len(self.vector_store.docstore._dict) == 0:
                    self.vector_store = None
                    for fname in ["index.faiss", "index.pkl"]:
                        fpath = os.path.join(self.index_dir, fname)
                        if os.path.exists(fpath):
                            os.remove(fpath)
                else:
                    self.vector_store.save_local(self.index_dir)

        # Delete extracted images folder from disk
        for ver in versions_to_delete:
            img_dir = os.path.join("vectorstore", "extracted_images", filename.replace(".", "_"), ver)
            if os.path.exists(img_dir):
                for img in os.listdir(img_dir):
                    os.remove(os.path.join(img_dir, img))
                os.rmdir(img_dir)

        # Update metadata JSON
        if version:
            if version in self.metadata[filename]["versions"]:
                del self.metadata[filename]["versions"][version]
            
            if not self.metadata[filename]["versions"]:
                del self.metadata[filename]
            else:
                if self.metadata[filename]["active_version"] == version:
                    self.metadata[filename]["active_version"] = list(self.metadata[filename]["versions"].keys())[-1]
        else:
            del self.metadata[filename]

        self._save_metadata()
        return True

    def set_active_version(self, filename, version):
        """Sets the active version for a document."""
        if filename in self.metadata and version in self.metadata[filename]["versions"]:
            self.metadata[filename]["active_version"] = version
            self._save_metadata()
            return True
        return False

    def extract_images(self, file_path, filename, version):
        """Extracts images from PDF pages and saves them locally."""
        images_dir = os.path.join("vectorstore", "extracted_images", filename.replace(".", "_"), version)
        os.makedirs(images_dir, exist_ok=True)
        
        extracted_paths = []
        try:
            pdf_reader = PdfReader(file_path)
            for page_idx, page in enumerate(pdf_reader.pages):
                for img_idx, image_file in enumerate(page.images):
                    image_name = f"page_{page_idx + 1}_img_{img_idx + 1}.png"
                    image_path = os.path.join(images_dir, image_name)
                    with open(image_path, "wb") as fp:
                        fp.write(image_file.data)
                    extracted_paths.append(image_path.replace("\\", "/"))
        except Exception as e:
            print(f"Image extraction warning: {e}")
            
        return extracted_paths

    def process_document(self, file_path, original_filename):
        """Processes a PDF: semantic chunking, indexing, and extracting AI highlights."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split into semantic chunks
        splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile"
        )
        chunks = splitter.split_documents(documents)

        # Determine version
        if original_filename not in self.metadata:
            self.metadata[original_filename] = {
                "active_version": "v1",
                "versions": {}
            }
            version_name = "v1"
        else:
            existing_versions = self.metadata[original_filename].get("versions", {})
            version_numbers = []
            for v in existing_versions.keys():
                try:
                    version_numbers.append(int(v.replace("v", "")))
                except ValueError:
                    pass
            next_num = max(version_numbers, default=0) + 1
            version_name = f"v{next_num}"
            self.metadata[original_filename]["active_version"] = version_name

        # Tag chunks with source and version metadata
        for chunk in chunks:
            chunk.metadata["source"] = original_filename
            chunk.metadata["version"] = version_name
            chunk.metadata["page"] = int(chunk.metadata.get("page", 0))

        # Add chunks to vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        os.makedirs(self.index_dir, exist_ok=True)
        self.vector_store.save_local(self.index_dir)

        # Extract images from PDF
        images = self.extract_images(file_path, original_filename, version_name)

        # Extract AI features: Summary, Keywords, Knowledge Graph, Charts
        summary_context = "\n".join([chunk.page_content for chunk in chunks[:4]])[:12000]
        
        summary = self._generate_summary(summary_context)
        keywords = self._extract_keywords(summary_context)
        knowledge_graph = self._extract_knowledge_graph(summary_context, original_filename)
        data_charts = self._extract_charts(summary_context)

        # Update metadata database
        self.metadata[original_filename]["versions"][version_name] = {
            "size_bytes": os.path.getsize(file_path),
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "keywords": keywords,
            "chunks_count": len(chunks),
            "images": images,
            "knowledge_graph": knowledge_graph,
            "data_charts": data_charts
        }

        self._save_metadata()
        return len(chunks)

    def query_stream(self, question, selected_files=None, k=5):
        """Streams the RAG answer for a user question."""
        if self.vector_store is None:
            yield "Please upload and process a document first."
            return

        start_retrieval = time.time()
        
        # Build active document-version target list
        target_files = selected_files if selected_files else list(self.metadata.keys())
        active_targets = []
        for fname in target_files:
            if fname in self.metadata:
                active_ver = self.metadata[fname].get("active_version")
                if active_ver:
                    active_targets.append((fname, active_ver))

        # Perform similarity search
        try:
            raw_results = self.vector_store.similarity_search(question, k=k*5)
            retrieved_docs = []
            for doc in raw_results:
                src = doc.metadata.get("source")
                ver = doc.metadata.get("version")
                if (src, ver) in active_targets:
                    retrieved_docs.append(doc)
                if len(retrieved_docs) >= k:
                    break
        except Exception:
            retrieved_docs = self.vector_store.similarity_search(question, k=k)

        retrieval_time = time.time() - start_retrieval

        # Format context
        context_parts = []
        for idx, doc in enumerate(retrieved_docs):
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0) + 1
            context_parts.append(f"--- Chunk {idx+1} | Source: {src} | Page: {page} ---\n{doc.page_content}")

        context_text = "\n\n".join(context_parts)
        system_prompt = (
            "You are an intelligent AI assistant.\n"
            "Answer the user's question ONLY using the retrieved context below.\n"
            "If the answer is not available in the context, reply with:\n"
            '"I don\'t know based on the uploaded document(s)."\n\n'
            f"Context:\n{context_text}"
        )

        prompt_messages = [
            ("system", system_prompt),
            ("human", question)
        ]

        self.last_query_metadata = {
            "retrieved_docs": retrieved_docs,
            "retrieval_time": retrieval_time,
            "generation_time": 0.0
        }

        # Stream response tokens
        start_generation = time.time()
        try:
            for chunk in self.llm.stream(prompt_messages):
                yield chunk.content
        except Exception as e:
            yield f"\n\nError generating response: {e}"

        self.last_query_metadata["generation_time"] = time.time() - start_generation

    def generate_suggested_questions(self, filenames):
        """Generates 3 relevant questions based on document summaries."""
        if not filenames:
            return [
                "What are the main topics discussed?",
                "Can you provide a summary of the key findings?",
                "What are the core conclusions of the document?"
            ]

        summaries = []
        for fname in filenames:
            if fname in self.metadata:
                active_ver = self.metadata[fname].get("active_version", "v1")
                ver_data = self.metadata[fname].get("versions", {}).get(active_ver, {})
                summaries.append(ver_data.get("summary", ""))

        combined = "\n".join(summaries)[:4000]
        if not combined.strip():
            return [
                "What are the main topics discussed?",
                "Can you provide a summary of the key findings?",
                "What are the core conclusions of the document?"
            ]

        try:
            prompt = (
                "Based on the following document summary, generate exactly 3 concise questions "
                "that a user might ask. Return ONLY a raw JSON list of strings, e.g. [\"Q1?\", \"Q2?\", \"Q3?\"]:\n\n"
                f"Summary:\n{combined}"
            )
            response = self.llm.invoke([("human", prompt)]).content.strip()
            clean_json = self._clean_json(response)
            questions = json.loads(clean_json)
            if isinstance(questions, list) and len(questions) >= 3:
                return [q.strip() for q in questions[:3]]
        except Exception:
            pass

        return [
            "What are the main topics discussed?",
            "Can you provide a summary of the key findings?",
            "What are the core conclusions of the document?"
        ]

    def compare_documents(self, doc_a, version_a, doc_b, version_b):
        """Generates a comparative analysis stream between two documents."""
        data_a = self.metadata.get(doc_a, {}).get("versions", {}).get(version_a, {})
        data_b = self.metadata.get(doc_b, {}).get("versions", {}).get(version_b, {})

        summary_a = data_a.get("summary", "No summary available.")
        summary_b = data_b.get("summary", "No summary available.")
        keywords_a = ", ".join(data_a.get("keywords", []))
        keywords_b = ", ".join(data_b.get("keywords", []))

        prompt = (
            "You are a helpful analyst.\n"
            "Compare the following two documents side-by-side. Outline key differences, "
            "similarities, and present a comparison table in Markdown.\n\n"
            f"Document 1: {doc_a} ({version_a})\nSummary: {summary_a}\nKeywords: {keywords_a}\n\n"
            f"Document 2: {doc_b} ({version_b})\nSummary: {summary_b}\nKeywords: {keywords_b}\n"
        )

        try:
            for chunk in self.llm.stream([("human", prompt)]):
                yield chunk.content
        except Exception as e:
            yield f"Error running comparison: {e}"

    # Private Helper Methods
    def _save_metadata(self):
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4)

    def _clean_json(self, text):
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _generate_summary(self, text):
        try:
            prompt = f"Write a concise 2-3 sentence summary of the following text:\n\n{text}"
            return self.llm.invoke([("human", prompt)]).content.strip()
        except Exception:
            return "Executive summary is currently unavailable."

    def _extract_keywords(self, text):
        try:
            prompt = f"Identify 5 key topics or keywords in this text. Return only a comma-separated list:\n\n{text}"
            result = self.llm.invoke([("human", prompt)]).content.strip()
            return [k.strip() for k in result.split(",") if k.strip()][:5]
        except Exception:
            return ["document", "pdf"]

    def _extract_knowledge_graph(self, text, filename):
        try:
            prompt = (
                "Extract main entities as 'nodes' and their connections as 'edges'. "
                'Return ONLY a valid JSON object in this format: {"nodes": [{"id": "A", "label": "A", "group": "Concept"}], "edges": [{"from": "A", "to": "B", "label": "relates"}]}\n\n'
                f"Text:\n{text}"
            )
            raw = self.llm.invoke([("human", prompt)]).content.strip()
            return json.loads(self._clean_json(raw))
        except Exception:
            return {
                "nodes": [
                    {"id": "Doc", "label": filename, "group": "Document"},
                    {"id": "RAG", "label": "RAG System", "group": "System"}
                ],
                "edges": [{"from": "Doc", "to": "RAG", "label": "Indexed"}]
            }

    def _extract_charts(self, text):
        try:
            prompt = (
                "Extract numerical statistics or metrics. "
                'Return ONLY a valid JSON list: [{"label": "Metric Name", "value": 100}]\n\n'
                f"Text:\n{text}"
            )
            raw = self.llm.invoke([("human", prompt)]).content.strip()
            data = json.loads(self._clean_json(raw))
            return data if isinstance(data, list) else []
        except Exception:
            return []