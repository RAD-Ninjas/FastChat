"""
A model worker that executes the model based on vLLM.

See documentations at docs/vllm_integration.md
"""

import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0,"/home/openvino/FastChat")

import argparse
import asyncio
import json
import os
import uuid
from threading import Thread, current_thread
from typing import List, Optional, Union

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from fastchat.serve.model_worker import (
    BaseModelWorker,
    logger,
    worker_id,
)
from fastchat.utils import get_context_length

from transformers import LlamaTokenizer, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM

app = FastAPI()

TOKEN = os.environ['HF_TOKEN']
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""


def build_inputs(history: list[tuple[str, str]],
                 query: str,
                 system_prompt=DEFAULT_SYSTEM_PROMPT) -> str:
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in history:
        texts.append(
            f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{query.strip()}')
    return ''.join(texts)


class LlamaModel():

    def __init__(self,
                 tokenizer_path,
                 device='GPU.1',
                 model_path='../ir_model_chat',
                 token=TOKEN) -> None:
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path,
                                                        trust_remote_code=True,
                                                        token=TOKEN)
        self.ov_model = OVModelForCausalLM.from_pretrained(model_path,
                                                           token=TOKEN,
                                                           compile=False,
                                                           use_cache=True)
        print(device)
        self.ov_model.to(device)
        self.ov_model.compile()

    def generate_iterate(self, queue, prompt: str, max_generated_tokens, top_k, top_p,
                         temperature):
        # Tokenize the user text.
        model_inputs = self.tokenizer(prompt, return_tensors="pt")

        # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
        # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
        streamer = TextIteratorStreamer(self.tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        generate_kwargs = dict(model_inputs,
                               streamer=streamer,
                               max_new_tokens=max_generated_tokens,
                               do_sample=True,
                               top_p=top_p,
                               temperature=float(temperature),
                               top_k=top_k,
                               eos_token_id=self.tokenizer.eos_token_id)

        print("Starting inference now...")
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(self.ov_model.generate,**generate_kwargs)
        print("Producing streaming outputs...")
        # Pull the generated text from the streamer, and update the model output.
        model_output = ""
        for new_text in streamer:
            print(f"Token: {new_text}")
            model_output += new_text
            queue.put_nowait(model_output)
        queue.put_nowait(None)

class OpenvinoWorker(BaseModelWorker):
    def __init__(
            self,
            controller_addr: str,
            worker_addr: str,
            worker_id: str,
            model_path: str,
            model_names: List[str],
            limit_worker_concurrency: int,
            no_register: bool,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: vLLM worker..."
        )
        self.tokenizer = ov_model.tokenizer
        self.context_len = 2048

        if not no_register:
            self.init_heart_beat()

    async def consumer(self, queue):
        print("Started consumer")
        while True:
            value = await queue.get()
            if value is None:
                break
            yield value
        print("Done.")

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True)

        # Handle stop_str
        if isinstance(stop_str, str) and stop_str != "":
            stop = [stop_str]
        elif isinstance(stop_str, list) and stop_str != []:
            stop = stop_str
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        queue = asyncio.Queue()
        print(f"CONTEXT: {context}")
        inputs = build_inputs([], context[60:]) # Remove useless prefix
        print(f"INPUTS: {inputs}")
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(ov_model.generate_iterate, queue, inputs,max_new_tokens, 20, top_p, temperature)

        async for output in self.consumer(queue):
            yield {"text": output, "error_code": 0, "usage": {}}
    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return x


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        pass

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    output = await worker.generate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--device", type=str, default="CPU")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="/home/openvino/models/llama-2/ir_model")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")

    args = parser.parse_args()
    args.model = "meta-llama/Llama-2-7b-chat-hf"

    print("*"*8+"LOADING MODEL... (this may take a while)"+"*"*8)
    ov_model = LlamaModel(args.model,
                          model_path=args.model_path,
                          token=os.environ.get("HF_TOKEN", ""),
                          device=args.device)
    print("*"*8+"MODEL LOADED!"+"*"*8)
    executor = ThreadPoolExecutor(max_workers=1)
    worker = OpenvinoWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
