import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from rag_engine import RAGPipeline
from rag_engine.sample_data import SAMPLE_DATASETS

# Global pipeline instance
pipeline_instance = RAGPipeline()

class RAGRequestHandler(BaseHTTPRequestHandler):

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json({"status": "ok", "message": "RAG Engine Server is running."})

        elif path == "/api/preset_datasets":
            datasets = [
                {"key": k, "name": k.replace("_", " ").title(), "doc_count": len(v)}
                for k, v in SAMPLE_DATASETS.items()
            ]
            self._send_json({"datasets": datasets})

        elif path == "/api/config":
            self._send_json({"config": pipeline_instance.config})

        elif path == "/api/chunks":
            chunks_data = [c.to_dict() for c in pipeline_instance.indexed_chunks]
            self._send_json({"chunks": chunks_data, "total": len(chunks_data)})

        else:
            self._send_json({"error": "Endpoint not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"

        try:
            body = json.loads(post_body)
        except Exception:
            body = {}

        if path == "/api/query":
            raw_query = body.get("query", "What is RAG?")
            response = pipeline_instance.query(raw_query)

            resp_dict = {
                "query": response.query,
                "answer": response.answer,
                "context": [d.to_dict() for d in response.context],
                "citations": response.citations,
                "metrics": response.metrics,
                "config_used": response.config_used,
                "step_logs": [
                    {
                        "step_number": l.step_number,
                        "step_name": l.step_name,
                        "description": l.description,
                        "input_summary": l.input_summary,
                        "output_summary": l.output_summary,
                        "details": l.details,
                        "execution_time_ms": l.execution_time_ms
                    }
                    for l in response.step_logs
                ]
            }
            self._send_json(resp_dict)

        elif path == "/api/ingest":
            dataset_key = body.get("dataset_key")
            custom_docs = body.get("documents")

            if dataset_key and dataset_key in SAMPLE_DATASETS:
                logs = pipeline_instance.load_preset_dataset(dataset_key)
            elif custom_docs:
                logs = pipeline_instance.ingest_documents(custom_docs)
            else:
                logs = pipeline_instance.load_preset_dataset("rag_architecture")

            self._send_json({
                "message": f"Successfully ingested documents.",
                "total_chunks": len(pipeline_instance.indexed_chunks),
                "indexing_logs": [
                    {
                        "step_number": l.step_number,
                        "step_name": l.step_name,
                        "output_summary": l.output_summary,
                        "execution_time_ms": l.execution_time_ms
                    }
                    for l in logs
                ]
            })

        elif path == "/api/config":
            new_cfg = body.get("config", {})
            for k, v in new_cfg.items():
                if k in pipeline_instance.config:
                    pipeline_instance.config[k] = v

            # Re-instantiate pipeline with updated config
            global pipeline_instance
            pipeline_instance = RAGPipeline(**pipeline_instance.config)
            self._send_json({"message": "Configuration updated.", "config": pipeline_instance.config})

        else:
            self._send_json({"error": "Endpoint not found"}, 404)

def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, RAGRequestHandler)
    print(f"RAG Engine API Server running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
