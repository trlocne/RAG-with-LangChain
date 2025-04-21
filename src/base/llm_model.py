import torch
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    pipeline
)
from langchain.llms.huggingface_pipeline import HuggingFacePipeline


def get_hf_llm(
    model_name: str = "meta-llama/Llama-3.1-8B",
    max_new_tokens: int = 10000,
    temperature: float = 0.7,
    **kwargs
):

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,                      
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs=kwargs
    )

    return llm
