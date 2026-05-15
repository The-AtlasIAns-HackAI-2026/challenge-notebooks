
import modal
import time
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel

app = modal.App("vllm-final-app")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.9.2",
        "transformers==4.53.2",
        "fastapi[standard]==0.115.0",
    )
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.95


@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=600,
)
@modal.concurrent(max_inputs=10)
class VLLMChatApp:

    @modal.enter()
    def load_model(self):
        from vllm import LLM

        self.llm = LLM(
            model=MODEL_NAME,
            dtype="float16",
            max_model_len=1024,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
        )

        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        self.errors = 0

    def format_chat(self, messages: List[Message]) -> str:
        prompt = ""

        for msg in messages:
            if msg.role == "system":
                prompt += f"<|system|>\n{msg.content}</s>\n"
            elif msg.role == "user":
                prompt += f"<|user|>\n{msg.content}</s>\n"
            elif msg.role == "assistant":
                prompt += f"<|assistant|>\n{msg.content}</s>\n"

        prompt += "<|assistant|>\n"
        return prompt

    @modal.method()
    def generate(self, request: ChatRequest) -> Dict:
        from vllm import SamplingParams

        try:
            start = time.time()

            prompt = self.format_chat(request.messages)

            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=["</s>", "<|user|>", "<|system|>"],
            )

            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0].outputs[0]

            latency = time.time() - start
            tokens_generated = len(output.token_ids)
            tokens_per_second = tokens_generated / latency if latency > 0 else 0

            self.total_requests += 1
            self.total_tokens += tokens_generated
            self.total_latency += latency

            return {
                "text": output.text.strip(),
                "tokens_generated": tokens_generated,
                "latency_seconds": round(latency, 3),
                "tokens_per_second": round(tokens_per_second, 2),
                "model": MODEL_NAME,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.errors += 1
            return {
                "error": str(e),
                "model": MODEL_NAME,
                "timestamp": datetime.now().isoformat(),
            }

    @modal.method()
    def stats(self) -> Dict:
        avg_latency = self.total_latency / self.total_requests if self.total_requests else 0
        avg_tokens = self.total_tokens / self.total_requests if self.total_requests else 0

        return {
            "model": MODEL_NAME,
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens,
            "average_latency_seconds": round(avg_latency, 3),
            "average_tokens_per_request": round(avg_tokens, 2),
            "errors": self.errors,
        }

    @modal.fastapi_endpoint(method="POST")
    def chat(self, request: ChatRequest) -> Dict:
        return self.generate.local(request)

    @modal.fastapi_endpoint(method="GET")
    def get_stats(self) -> Dict:
        return self.stats.local()

    @modal.fastapi_endpoint(method="GET")
    def ui(self):
        from fastapi.responses import HTMLResponse

        chat_url = "https://hbouanane--vllm-final-app-vllmchatapp-chat.modal.run" #REPLACE WITH YOUR OWN URL
        stats_url = "https://hbouanane--vllm-final-app-vllmchatapp-get-stats.modal.run" #REPLACE WITH YOUR OWN URL

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>vLLM Chat UI</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f4f4f4;
        }}
        .container {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        }}
        textarea {{
            width: 100%;
            height: 140px;
            padding: 12px;
            font-size: 16px;
            box-sizing: border-box;
        }}
        button {{
            margin-top: 12px;
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
        }}
        #response {{
            margin-top: 20px;
            background: #efefef;
            padding: 15px;
            border-radius: 8px;
            white-space: pre-wrap;
            min-height: 120px;
        }}
        #stats {{
            margin-top: 20px;
            background: #dfefff;
            padding: 15px;
            border-radius: 8px;
            white-space: pre-wrap;
        }}
    </style>
</head>

<body>
    <div class="container">
        <h1>vLLM Chat UI</h1>

        <textarea id="prompt" placeholder="Ask something..."></textarea>
        <br>

        <button onclick="sendMessage()">Send</button>
        <button onclick="loadStats()">Load Stats</button>

        <div id="response"></div>
        <div id="stats"></div>
    </div>

<script>
async function sendMessage() {{

    const prompt = document.getElementById("prompt").value;
    const responseDiv = document.getElementById("response");

    responseDiv.innerText = "Generating...";

    const res = await fetch("{chat_url}", {{
        method: "POST",
        headers: {{
            "Content-Type": "application/json"
        }},
        body: JSON.stringify({{
            messages: [
                {{
                    role: "system",
                    content: "You are a helpful AI assistant."
                }},
                {{
                    role: "user",
                    content: prompt
                }}
            ],
            max_tokens: 150,
            temperature: 0.7,
            top_p: 0.95
        }})
    }});

    const data = await res.json();

    if (data.text) {{
        responseDiv.innerText =
            data.text +
            "\\n\\n--- Metrics ---" +
            "\\nTokens: " + data.tokens_generated +
            "\\nLatency: " + data.latency_seconds + "s" +
            "\\nTokens/sec: " + data.tokens_per_second +
            "\\nModel: " + data.model;
    }} else {{
        responseDiv.innerText = "Error: " + JSON.stringify(data);
    }}
}}

async function loadStats() {{

    const statsDiv = document.getElementById("stats");

    statsDiv.innerText = "Loading stats...";

    const res = await fetch("{stats_url}");
    const data = await res.json();

    statsDiv.innerText =
        "=== Aggregate Statistics ===\\n\\n" +
        "Model: " + data.model + "\\n" +
        "Total Requests: " + data.total_requests + "\\n" +
        "Total Tokens: " + data.total_tokens_generated + "\\n" +
        "Average Latency: " + data.average_latency_seconds + "s\\n" +
        "Average Tokens/Request: " + data.average_tokens_per_request + "\\n" +
        "Errors: " + data.errors;
}}
</script>

</body>
</html>
        """

        return HTMLResponse(html)


@app.local_entrypoint()
def main():

    model = VLLMChatApp()

    request = ChatRequest(
        messages=[
            Message(role="system", content="You are a helpful AI assistant."),
            Message(role="user", content="Explain machine learning simply."),
        ],
        max_tokens=120,
        temperature=0.7,
    )

    response = model.generate.remote(request)

    print("\\nChat response:")
    print(response)

    stats = model.stats.remote()

    print("\\nStats:")
    print(stats)
