import os
import logging
import re
import json
from typing import List, Dict, Any, TypedDict

import fitz  # PyMuPDF
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from pinecone import Pinecone
from langgraph.graph import StateGraph, END

from config import config

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        logger.info("Initializing RAGService...")
        config.validate()

        self.llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

        # Pinecone setup
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": config.PINECONE_ENVIRONMENT}}
            )

        self.vector_store = PineconeVectorStore(index_name=config.PINECONE_INDEX_NAME, embedding=self.embeddings)
        self.ensemble_retriever = None
        self.documents = []
        self.section_map = {}
        # ADD: Separate storage for table documents
        self.table_documents = []
        
        self.graph = self._create_graph()
        logger.info("RAG Service initialized.")

    def ingest_document(self, file_path: str):
        """Enhanced document ingestion with better table handling."""
        logger.info(f"Ingesting: {file_path}")
        doc = fitz.open(file_path)
        
        all_chunks = []
        full_text = ""

        # Extract text and tables
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            if not page_text.strip():
                page_text = "\n".join([block[4] for block in page.get_text("blocks")])
            
            full_text += f"\n--- PAGE {page_num + 1} ---\n" + page_text

            # ENHANCED: Better table extraction with context
            for table in page.find_tables():
                try:
                    table_md = table.to_markdown(clean=True)
                    if len(table_md.strip()) > 50:
                        # ENHANCED: Add context around table
                        table_bbox = table.bbox
                        context_text = self._extract_table_context(page, table_bbox)
                        
                        table_content = f"Context: {context_text}\n\nTable:\n{table_md}"
                        
                        table_doc = Document(
                            page_content=table_content,
                            metadata={
                                "source": file_path,
                                "page": page_num + 1,
                                "type": "table",
                                "priority": 4,  # Higher priority for tables
                                "table_headers": self._extract_table_headers(table_md),
                                "context": context_text
                            }
                        )
                        all_chunks.append(table_doc)
                        self.table_documents.append(table_doc)  # Store separately
                except:
                    continue

        doc.close()

        # Simple section splitting - focus on clear patterns
        sections = self._split_into_sections(full_text, file_path)
        all_chunks.extend(sections)

        # Chunk large sections
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # Smaller chunks for precision
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )

        final_chunks = []
        for chunk in all_chunks:
            if len(chunk.page_content) > 400 and chunk.metadata.get("type") != "table":  # Don't split tables
                sub_chunks = text_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        # Add cross-references
        self._add_simple_cross_refs(final_chunks)
        
        self.documents = final_chunks
        self._build_section_map()
        self._setup_retrievers()
        
        logger.info(f"Created {len(final_chunks)} chunks, {len(self.table_documents)} tables")

    # NEW: Extract context around tables
    def _extract_table_context(self, page, table_bbox):
        """Extract text context around a table."""
        try:
            # Get text blocks
            blocks = page.get_text("blocks")
            context_parts = []
            
            for block in blocks:
                block_bbox = block[:4]  # x0, y0, x1, y1
                # Check if block is near the table (above or below)
                if (block_bbox[3] < table_bbox[1] and table_bbox[1] - block_bbox[3] < 100) or \
                   (block_bbox[1] > table_bbox[3] and block_bbox[1] - table_bbox[3] < 100):
                    text = block[4].strip()
                    if text and len(text) > 10:
                        context_parts.append(text)
            
            return " ".join(context_parts[:2])  # Limit context
        except:
            return ""

    # NEW: Extract table headers for better matching
    def _extract_table_headers(self, table_md):
        """Extract headers from markdown table."""
        try:
            lines = table_md.split('\n')
            if len(lines) > 0:
                header_line = lines[0]
                headers = [h.strip().strip('|') for h in header_line.split('|') if h.strip()]
                return headers
        except:
            pass
        return []

    def _split_into_sections(self, text, file_path):
        """Simple, reliable section splitting."""
        sections = []
        
        # Basic patterns that actually work
        patterns = [
            r'^([A-Z][A-Z\s]{5,})\s*$',  # ALL CAPS HEADINGS
            r'^(\d+\.\s+[A-Z][^.]*?)$',   # "1. SECTION NAME"
            r'^(ARTICLE\s+[IVXLC\d]+[^\n]*?)$',
            r'^(SECTION\s+[IVXLC\d]+[^\n]*?)$'
        ]
        
        current_section = ""
        current_title = "Introduction"
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            is_header = False
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Save previous section
                    if current_section.strip():
                        sections.append(Document(
                            page_content=current_section.strip(),
                            metadata={
                                "source": file_path,
                                "section_title": current_title,
                                "type": "text",
                                "priority": self._get_section_priority(current_title)
                            }
                        ))
                    
                    current_title = line
                    current_section = ""
                    is_header = True
                    break
            
            if not is_header:
                current_section += line + "\n"
        
        # Add final section
        if current_section.strip():
            sections.append(Document(
                page_content=current_section.strip(),
                metadata={
                    "source": file_path,
                    "section_title": current_title,
                    "type": "text",
                    "priority": self._get_section_priority(current_title)
                }
            ))
        
        return sections

    def _get_section_priority(self, title):
        """Simple priority assignment."""
        title_upper = title.upper()
        if any(word in title_upper for word in ["DEFINITION", "MEANING"]):
            return 2
        elif any(word in title_upper for word in ["COVERAGE", "BENEFIT"]):
            return 2
        elif any(word in title_upper for word in ["EXCLUSION", "LIMIT"]):
            return 2
        return 1

    def _add_simple_cross_refs(self, chunks):
        """Simplified cross-reference detection."""
        section_titles = {chunk.metadata.get("section_title", "") for chunk in chunks}
        section_titles = {title for title in section_titles if title and len(title) > 5}
        
        for chunk in chunks:
            refs = []
            content_lower = chunk.page_content.lower()
            
            # Find section references
            for title in section_titles:
                if title != chunk.metadata.get("section_title", ""):
                    # Check for title or key words in content
                    title_words = [word for word in title.lower().split() if len(word) > 3]
                    if any(word in content_lower for word in title_words[:2]):  # First 2 key words
                        refs.append(title)
            
            if refs:
                chunk.metadata["cross_refs"] = refs[:3]  # Limit to 3

    def _build_section_map(self):
        """Build mapping for cross-linking."""
        self.section_map = {}
        for chunk in self.documents:
            title = chunk.metadata.get("section_title", "")
            if title:
                if title not in self.section_map:
                    self.section_map[title] = []
                self.section_map[title].append(chunk)

    def _setup_retrievers(self):
        """Setup retrievers with error handling."""
        try:
            bm25 = BM25Retriever.from_documents(self.documents, k=8)
            self.vector_store.add_documents(self.documents)
            vector = self.vector_store.as_retriever(search_kwargs={"k": 8})
            
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25, vector],
                weights=[0.4, 0.6]  # Favor vector search slightly
            )
        except Exception as e:
            logger.error(f"Retriever setup failed: {e}")

    # LangGraph nodes - enhanced for tables

    def classify_query(self, state):
        """Enhanced classification for tabular queries."""
        question = state["question"].lower()
        
        # ENHANCED: Check for tabular/numerical indicators
        table_keywords = ["table", "chart", "rate", "amount", "list", "schedule", "premium", 
                         "deductible", "limit", "maximum", "minimum", "percentage", "%", "$"]
        numerical_keywords = ["how much", "amount", "cost", "price", "rate", "%", "percent", 
                            "dollar", "maximum", "minimum", "limit"]
        
        if any(word in question for word in table_keywords) or \
           any(word in question for word in numerical_keywords):
            category = "TABULAR"
        elif any(word in question for word in ["what is", "define", "meaning", "means"]):
            category = "DEFINITION"
        elif any(word in question for word in ["covered", "include", "benefit"]):
            category = "COVERAGE"
        elif any(word in question for word in ["not covered", "exclude", "limitation"]):
            category = "EXCLUSION"
        elif any(word in question for word in ["how to", "process", "procedure", "step"]):
            category = "PROCEDURE"
        else:
            category = "GENERAL"
        
        return {"category": category}

    def retrieve(self, state):
        """Enhanced retrieval with table-specific logic."""
        question = state["question"]
        category = state["category"]
        
        # Primary retrieval
        docs = self.ensemble_retriever.invoke(question)
        
        # ENHANCED: Table-specific retrieval
        if category == "TABULAR":
            # Prioritize table documents
            table_docs = []
            question_words = set(question.lower().split())
            
            for table_doc in self.table_documents:
                # Check if question words match table headers or context
                headers = table_doc.metadata.get("table_headers", [])
                context = table_doc.metadata.get("context", "").lower()
                
                header_match = any(any(word in header.lower() for word in question_words) 
                                 for header in headers)
                context_match = any(word in context for word in question_words if len(word) > 3)
                
                if header_match or context_match:
                    table_docs.append(table_doc)
            
            # Add top matching table docs to front
            docs = table_docs[:3] + docs
        
        # Add category-specific docs
        elif category == "DEFINITION":
            definition_docs = [d for d in self.documents if d.metadata.get("priority", 1) >= 2 and "definit" in d.page_content.lower()]
            docs.extend(definition_docs[:2])
        
        # Add cross-referenced docs (limited)
        cross_docs = []
        for doc in docs[:3]:  # Only top 3
            refs = doc.metadata.get("cross_refs", [])
            for ref in refs[:2]:  # Max 2 refs per doc
                if ref in self.section_map:
                    cross_docs.extend(self.section_map[ref][:1])  # 1 chunk per ref
        
        # Combine and deduplicate
        all_docs = docs + cross_docs
        unique_docs = self._dedupe_simple(all_docs)
        
        return {"documents": unique_docs[:10]}  # Limit total docs

    def _dedupe_simple(self, docs):
        """Simple deduplication."""
        seen = set()
        unique = []
        for doc in docs:
            key = hash(doc.page_content[:50])  # First 50 chars
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def rerank(self, state):
        """Enhanced reranking with table priority."""
        question = state["question"]
        docs = state["documents"]
        category = state["category"]
        
        if not docs:
            return {"documents": []}
        
        # ENHANCED: Boost table documents for tabular queries
        if category == "TABULAR":
            for doc in docs:
                if doc.metadata.get("type") == "table":
                    doc.metadata['score'] = doc.metadata.get('score', 0.5) + 0.3  # Boost tables
        
        # Rerank only if we have many docs
        if len(docs) > 5:
            pairs = [(question, doc.page_content) for doc in docs]
            scores = self.reranker.predict(pairs)
            
            for i, doc in enumerate(docs):
                current_score = doc.metadata.get('score', 0)
                doc.metadata['score'] = max(current_score, float(scores[i]))
            
            docs.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
        
        return {"documents": docs[:5]}  # Top 5 only

    def generate_answer(self, state):
        """Enhanced answer generation for tabular data."""
        question = state["question"]
        docs = state["documents"]
        category = state["category"]
        
        if not docs:
            return {"answer": "No relevant information found."}
        
        # Build minimal context
        context_parts = []
        has_table = False
        
        for i, doc in enumerate(docs[:3]):  # Max 3 docs for context
            source = doc.metadata.get("section_title", f"Source {i+1}")
            content = doc.page_content
            
            # ENHANCED: Handle tables differently
            if doc.metadata.get("type") == "table":
                has_table = True
                # Keep full table content for tabular queries
                if category == "TABULAR":
                    context_parts.append(f"[Table from {source}]: {content}")
                else:
                    content = content[:300] + "..." if len(content) > 300 else content
                    context_parts.append(f"[Table from {source}]: {content}")
            else:
                content = content[:300] + "..." if len(content) > 300 else content
                context_parts.append(f"[{source}]: {content}")
        
        context = "\n\n".join(context_parts)
        
        # ENHANCED: Category-specific prompts
        if category == "TABULAR":
            instruction = """Extract specific values, rates, or data from the tables. 
            Present numerical information clearly. If comparing values, show the comparison clearly."""
        elif category == "DEFINITION":
            instruction = "Provide a clear, concise definition in 1-2 sentences."
        elif category == "NUMERICAL":
            instruction = "Extract and state the specific number/amount. Be precise."
        elif category == "COVERAGE":
            instruction = "Clearly state what IS covered in 2-3 sentences."
        elif category == "EXCLUSION":
            instruction = "Clearly state what is NOT covered in 2-3 sentences."
        else:
            instruction = "Answer directly and concisely in 2-3 sentences."

        prompt = ChatPromptTemplate.from_template(
            f"""Based on the context below, answer this question: {{question}}

{instruction} Do not add extra explanations or repeat the question.

Context:
{{context}}

Answer:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context})
        
        return {"answer": answer.strip()}

    def check_confidence(self, state):
        """Enhanced confidence check considering tables."""
        docs = state["documents"]
        category = state["category"]
        
        if not docs:
            return "handle_missing"
        
        # Simple heuristics
        top_score = docs[0].metadata.get('score', 0.5)
        has_priority = any(doc.metadata.get('priority', 1) > 1 for doc in docs)
        has_exact_match = any(word in docs[0].page_content.lower() for word in state["question"].lower().split())
        has_table = any(doc.metadata.get('type') == 'table' for doc in docs)
        
        confidence = top_score
        if has_priority:
            confidence += 0.1
        if has_exact_match:
            confidence += 0.1
        # ENHANCED: Boost confidence for tabular queries with tables
        if category == "TABULAR" and has_table:
            confidence += 0.2
        
        return "generate" if confidence > 0.3 else "handle_missing"

    def handle_missing_info(self, state):
        """Concise fallback."""
        return {"answer": "I couldn't find specific information about this in the document."}

    def _create_graph(self):
        """Simplified graph."""
        class GraphState(TypedDict):
            question: str
            category: str
            documents: List[Document]
            answer: str

        workflow = StateGraph(GraphState)
        
        workflow.add_node("classify", self.classify_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("handle_missing", self.handle_missing_info)

        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_conditional_edges(
            "rerank",
            self.check_confidence,
            {"generate": "generate", "handle_missing": "handle_missing"}
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("handle_missing", END)

        return workflow.compile()

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Main interface - simplified."""
        if not self.ensemble_retriever:
            return {"error": "No document ingested."}
        
        try:
            result = self.graph.invoke({"question": question})
            return {"answer": result.get('answer', 'No answer generated.')}
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"error": f"Query failed: {str(e)}"}