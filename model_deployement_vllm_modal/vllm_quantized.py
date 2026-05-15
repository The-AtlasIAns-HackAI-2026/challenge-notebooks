"""
Deploy quantized model with vLLM on Modal
"""

import modal

app = modal.App("vllm-quantized")

MODEL_NAME = "TheBloke/Llama-2-7B-Chat-AWQ"

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.9.2",
        "transformers==4.53.2",
    )
)

@app.cls(
    image=vllm_image,
    gpu="T4",
    scaledown_window=600,
)
class QuantizedVLLM:
    @modal.enter()
    def load(self):
        from vllm import LLM

        self.llm = LLM(
            model=MODEL_NAME,
            quantization="awq",
            dtype="float16",
            max_model_len=1024,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
        )

        print("Quantized model loaded!")

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text


@app.local_entrypoint()
def main():
    model = QuantizedVLLM()

    prompt = "Explain quantization in simple terms."
    result = model.generate.remote(prompt, max_tokens=150)

    print("Prompt:", prompt)
    print("Output:", result)
