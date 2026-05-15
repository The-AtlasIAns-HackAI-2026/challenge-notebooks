import modal
from pydantic import BaseModel

app = modal.App("vllm-api")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.9.2",
        "transformers==4.53.2",
        "fastapi[standard]==0.115.0",
    )
)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

class CompletionResponse(BaseModel):
    text: str
    tokens_used: int
    model: str

@app.cls(
    image=vllm_image,
    gpu="T4",
    scaledown_window=600,
)
@modal.concurrent(max_inputs=10)
class VLLMServer:
    @modal.enter()
    def startup(self):
        from vllm import LLM

        self.llm = LLM(
            model=MODEL_NAME,
            dtype="float16",
            max_model_len=1024,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
        )

    @modal.method()
    def generate(self, request: CompletionRequest) -> CompletionResponse:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        outputs = self.llm.generate([request.prompt], params)
        output = outputs[0].outputs[0]

        return CompletionResponse(
            text=output.text,
            tokens_used=len(output.token_ids),
            model=MODEL_NAME,
        )

    @modal.fastapi_endpoint(method="POST")
    def api_generate(self, request: CompletionRequest) -> CompletionResponse:
        return self.generate.local(request)

@app.local_entrypoint()
def test_api():
    server = VLLMServer()

    request = CompletionRequest(
        prompt="Write a haiku about AI:",
        max_tokens=100,
        temperature=0.8,
    )

    response = server.generate.remote(request)
    print("Generated:", response.text)
    print("Tokens used:", response.tokens_used)
