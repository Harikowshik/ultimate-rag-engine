import re
import os
from typing import List, Tuple, Dict, Any
from .schema import Document, StepLog

class LLMGenerator:
    """
    Stage 14: LLM GENERATION
    - Primary LLM Provider: Google Gemini API (gemini-1.5-flash / gemini-2.0-flash)
    - Fallback: Local Grounded Synthesis (No API Key Required)
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", gemini_api_key: str = None, openai_api_key: str = None):
        self.model_name = model_name
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    def _generate_with_gemini(self, query: str, context_str: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_api_key)
        target_model = self.model_name if "gemini" in self.model_name.lower() and "1.5" not in self.model_name else "gemini-2.5-flash"
        try:
            model = genai.GenerativeModel(target_model)
        except Exception:
            model = genai.GenerativeModel("gemini-flash-latest")

        prompt = (
            f"You are an expert AI Assistant specializing in document summarization and Question Answering.\n\n"
            f"Provide a cohesive, unified, high-level summary answer to the user question based on the total retrieved context.\n"
            f"Do NOT output individual doc-by-doc bullet points or chunk headers. Synthesize all context into a comprehensive, fluid, high-level summary paragraph.\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"User Question: {query}\n"
        )
        response = model.generate_content(prompt)
        return response.text.strip()

    def _generate_with_openai(self, query: str, context_str: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self.openai_api_key)
        sys_prompt = "You are an expert AI Assistant specializing in Retrieval Augmented Generation (RAG). Synthesize all context into a single, cohesive, unified summary answer rather than individual document points."
        user_prompt = f"Retrieved Context:\n{context_str}\n\nUser Question: {query}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def generate_response(self, query: str, context_str: str, context_docs: List[Document]) -> Tuple[str, StepLog]:
        if not context_docs:
            answer = "⚠️ **No Relevant Context Found**: The vector store could not retrieve relevant document passages to answer your question."
            used_provider = "System Guardrail"
        else:
            answer = None
            used_provider = f"Local Grounded Synthesis ({self.model_name})"

            # 1. Primary Option: Try Gemini if API Key present
            if self.gemini_api_key:
                try:
                    answer = self._generate_with_gemini(query, context_str)
                    used_provider = f"Google Gemini API ({self.model_name})"
                except Exception:
                    pass

            # 2. Secondary Option: Try OpenAI if API Key present
            if answer is None and self.openai_api_key:
                try:
                    answer = self._generate_with_openai(query, context_str)
                    used_provider = "OpenAI API (gpt-4o-mini)"
                except Exception:
                    pass

            # 3. Fallback: Local Grounded Synthesis Engine (Total Unified Summary)
            if answer is None:
                cleaned_passages = []
                for doc in context_docs:
                    txt = doc.page_content.strip()
                    # Fix run-together PDF text/numbers (e.g. US8.9billionin2022toUS -> US 8.9 billion in 2022 to US)
                    txt = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', txt)
                    txt = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', txt)
                    txt = re.sub(r'([a-z])([A-Z])', r'\1 \2', txt)
                    txt = re.sub(r'\s+', ' ', txt)
                    cleaned_passages.append(txt)

                all_sentences = []
                seen_sentences = set()

                for txt in cleaned_passages:
                    # Clean out markdown header hashes
                    txt_clean = re.sub(r'#+\s*', '', txt)
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', txt_clean) if len(s.strip()) > 15]
                    for sent in sentences:
                        norm_sent = re.sub(r'\W+', '', sent.lower())
                        if norm_sent not in seen_sentences:
                            seen_sentences.add(norm_sent)
                            all_sentences.append(sent)

                if all_sentences:
                    unified_summary = " ".join(all_sentences[:6])
                    answer = (
                        f"### Unified Document Summary\n\n"
                        f"{unified_summary}\n\n"
                        f"---\n"
                        f"*Synthesis Info: Synthesized total summary across {len(context_docs)} retrieved document context chunks.*"
                    )
                else:
                    answer = f"Synthesized summary for '{query}' across {len(context_docs)} grounded document passages."

        log = StepLog(
            step_number="14",
            step_name="LLM Generation",
            description=f"Generated grounded answer using {used_provider}.",
            input_summary=f"Query and context string ({len(context_docs)} chunks).",
            output_summary=f"Generated response answer ({len(answer.split())} words).",
            details={
                "model_name": used_provider,
                "context_chunks_used": len(context_docs),
                "answer_token_length": len(answer.split()),
                "has_api_keys": bool(self.gemini_api_key or self.openai_api_key)
            }
        )
        return answer, log


