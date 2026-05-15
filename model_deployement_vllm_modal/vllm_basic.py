import modal

app = modal.App("vllm-basic-example")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.9.2",
        "transformers==4.53.2",
        "accelerate",
    )
)

@app.cls(
    image=vllm_image,
    gpu="T4",
    scaledown_window=300,
)
class VLLMModel:
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

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text


@app.local_entrypoint()
def main():
    model = VLLMModel()

    prompt = "Explain machine learning in simple terms."
    result = model.generate.remote(prompt)

    print("Prompt:", prompt)
    print("Generated:", result)
