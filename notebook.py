import boto3, json, sagemaker
from typing import Dict
from langchain import LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
MY_HUGGING_FACE_TOKEN = "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"

role = sagemaker.get_execution_role()

# A read-only token is required for gated models
HUGGING_FACE_HUB_TOKEN = MY_HUGGING_FACE_TOKEN

hub = {
    'HF_MODEL_ID': 'mistralai/Mistral-7B-Instruct-v0.3',
    'SM_NUM_GPUS': '1',
    'HUGGING_FACE_HUB_TOKEN': HUGGING_FACE_HUB_TOKEN,
}

huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface", version="2.0"),
    env=hub,
    role=role 
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    container_startup_health_check_timeout=300,
  )


model_kwargs = {"max_new_tokens": 512, "top_p": 0.8, "temperature": 0.8}

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps(
            # Mistral prompt, see https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
            {"inputs": f"<s>[INST] {prompt} [/INST]", "parameters": {**model_kwargs}}
        )
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        splits = response_json[0]["generated_text"].split("[/INST] ")
        return splits[0]


content_handler = ContentHandler()


endpoint_name = predictor.endpoint_name


sm_client = boto3.client("sagemaker-runtime") # needed for AWS credentials

llm = SagemakerEndpoint(
    endpoint_name=endpoint_name,
    model_kwargs=model_kwargs,
    content_handler=content_handler,
    client=sm_client,
)



system_prompt = """
As a helpful energy specialist, please answer the question, focusing on numerical data.
Don't invent facts. If you can't provide a factual answer, say you don't know what the answer is.
"""

prompt = PromptTemplate.from_template(system_prompt + "{content}")

llm_chain = LLMChain(llm=llm, prompt=prompt)


question = "What is the trend for solar investments in China in 2023 and beyond?"

query = f"question: {question}"

answer = llm_chain.run({query})
#print(answer)
answer.split('[/INST]')[1]