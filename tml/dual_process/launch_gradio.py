from datetime import datetime
import glob
import gradio as gr
import json
import numpy as np
import pytz
from omegaconf import OmegaConf
import os
from PIL import Image
import random
import string
import time
import torch

from dual_process import dig_helpers, dig_pipeline, dig_viz, gpt_helpers

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# Setup pipe and vlm
PIPE_DEFAULT, VLM_DEFAULT = "schnell", "idefics2"
DEFAULT_CONFIGS = ["configs/base.yaml", "configs/app/app.yaml"]
PIPE_CONFIGS = [os.path.basename(x).split(".")[0] for x in glob.glob("configs/pipe/*.yaml")]
VLM_CONFIGS = [os.path.basename(x).split(".")[0] for x in glob.glob("configs/vlm/*.yaml")]

def compile_config(pipe_name, vlm_name):
    config_files = [
        f"configs/pipe/{pipe_name}.yaml",
        f"configs/vlm/{vlm_name}.yaml"
    ] + DEFAULT_CONFIGS
    return dig_helpers.load_config(config_files)

config = compile_config(PIPE_DEFAULT, VLM_DEFAULT)
old_config = config
pipe = dig_helpers.load_pipe(**config["pipe_kwargs"])
vlm, vlm_processor = dig_helpers.load_vlm(**config["vlm_kwargs"])

# Setup OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None
    print("Warning: OPENAI_API_KEY environment variable not set. OpenAI client will not be initialized.")

# Setup gradio logic
executor = ThreadPoolExecutor(max_workers=2)
iteration_checker = None

# ====================
#      Results
# ====================
def clear_page():
    iteration_checker.break_iter = True
    gr.Warning('🚨 Early stopping generation! Wait another minute for the final eval images or reload the page.')
    return None

def open_image(file_path):
    for _ in range(100):
        try:
            with Image.open(file_path) as image:
                image.load()
                return image.copy() 
        except Exception as e:
            print(e)

def show_results(qa_folder):
    global config
    final_eval = sorted(glob.glob(f"{qa_folder}/eval*"))[-1]
    results = json.load(open(f"{qa_folder}/results.json"))
    step = os.path.basename(final_eval).split("_")[-1].split(".")[0]
    scores = results["metas_probs"][f"lora_{step}"]
    grid = open_image(final_eval)
    size = grid.size[0]
    dim = (config["eval_kwargs"]["eval_n"], len(config["eval_kwargs"]["eval_weights"]))

    scores = np.array(scores).reshape((*dim, len(scores[0])))
    split = dig_viz.split_grid(grid, dim)

    viz = []
    for i, row in enumerate(split):
        format_score = lambda _scores: f"avg: {round(_scores.mean().item(), 2)}" + "\n" + ",".join([str(round(x, 2)) for x in _scores])
        row_scores = dig_viz.view_images([dig_viz.display_text(size // dim[1], 50, f"score={format_score(scores[i][j])}") for j in range(dim[1])], offset_ratio=0)
        row_images = dig_viz.view_images(row)
        row_images = dig_viz.resize_to_side(row_images, row_scores.size[0], "max")
        viz.extend([row_scores, row_images])
    grid = dig_viz.stack_images(viz)
    return grid

def monitor_results(qa_folder):
    while not os.path.exists(f"{qa_folder}/results.json"):
        readouts = sorted(glob.glob(f"{qa_folder}/readout/*"))
        if readouts:
            image = open_image(readouts[-1])
            yield image
        time.sleep(5)
    image = show_results(qa_folder)
    yield image

# ====================
#     Optimization
# ====================
def create_folder():
    global config
    save_name = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    pst = pytz.timezone('America/Los_Angeles')
    save_time = datetime.now(pst).strftime("%Y%m%d_%H%M%S")
    save_name = f"{save_time}_{save_name}"
    qa_folder = f"{config['save_folder']}/{save_name}"
    if not os.path.exists(qa_folder):
        os.makedirs(qa_folder, exist_ok="True")
    return qa_folder

def filter_qa_pairs(config, qa_pairs):
    filtered_qa_pairs = []
    qa_error_msg = None
    for question, answer in qa_pairs:
        if not question:
            continue
        if not answer:
            qa_error_msg = "Make sure every question has a corresponding answer!"
        question_postfix = config.get("question_postfix", "")
        yn_question = "Yes" in question or "No" in question or "Yes" in question_postfix or "No" in question_postfix
        yn_answer = answer == "Yes" or answer == "No"
        if yn_answer and not yn_question:
            qa_error_msg = "Make sure Yes/No questions end with the postfix ' Answer with Yes or No.'"
        if yn_question and not yn_answer:
            qa_error_msg = "The key 'question_postfix' in the config may be appending ' Answer with Yes or No.' to every question. Either override this key or change the answer to Yes/No."
        filtered_qa_pairs.append([question, answer])
    return filtered_qa_pairs, qa_error_msg

def create_input_to_edit(prompt, qa_pairs, ref_image=None, overlay_mode=None):
    global config, pipe, vlm, vlm_processor
    if config.get("prefix") is not None:
        gr.Warning(f"Using custom VLM input {config['prefix']}")
    if ref_image is not None:
        gr.Warning("Using overlaid reference image! Make sure your questions are worded appropriately.")
        ref_images = [ref_image]
    else:
        ref_images = []
    qa_pairs, qa_error_msg = filter_qa_pairs(config, qa_pairs)
    qa_pairs = [
        {
            "question": question, 
            "answer": answer,
            "ref_images": ref_images,
            "overlay_mode": overlay_mode
        } for question, answer in qa_pairs
    ]
    edit = dig_pipeline.create_edit(pipe, vlm, vlm_processor, config, qa_pairs, prompt)
    return edit, qa_error_msg

def call_optimize(edit, prompt, qa_pairs, qa_folder, params):
    global config, pipe, iteration_checker
    iteration_checker = dig_pipeline.IterationChecker()
    dig_pipeline.optimize(
        pipe, 
        edit, 
        config["opt_kwargs"], 
        config["generator_kwargs"], 
        [params], 
        prompt=prompt, 
        save_folder=qa_folder, 
        config=config, 
        eval_interactive=config.get("eval_interactive", 1),
        iteration_checker=iteration_checker
    )

def optimize(prompt, qa_pairs, ref_image, overlay_mode):
    global config
    ref_image = preprocess_ref_image(ref_image)

    # Validate qa_pairs
    if not qa_pairs or all([q == "" for q, a in qa_pairs]):
        gr.Warning("Enter question answer pairs!")
        return None
    
    edit, qa_error_msg = create_input_to_edit(prompt, qa_pairs, ref_image, overlay_mode)
    if qa_error_msg:
        gr.Warning(qa_error_msg)
        return None

    # Create save_folder
    qa_folder = create_folder()
    params = dig_helpers.create_lora(pipe, **config["lora_kwargs"])
    OmegaConf.save(config, f"{qa_folder}/config.yaml")
    json.dump({"prompt": prompt, "qa": qa_pairs}, open(f"{qa_folder}/qa.json", "w"))
    
    if ref_image is not None:
        ref_image.save(f"{qa_folder}/ref_image.png")

    # call_optimize(edit, prompt, qa_pairs, qa_folder, params)
    executor.submit(call_optimize, edit, prompt, qa_pairs, qa_folder, params)
    for image in monitor_results(qa_folder):
        yield image

def vanilla(prompt, qa_pairs, ref_image, overlay_mode):
    global config, pipe
    ref_image = preprocess_ref_image(ref_image)
    edit, _ = create_input_to_edit(prompt, qa_pairs, ref_image, overlay_mode)
    grid = dig_pipeline.visualize_vanilla_results(
        pipe, 
        config["generator_kwargs"], 
        prompt, 
        edit,
        **config["vanilla_kwargs"]
    )
    return grid

# ====================
#       Config
# ====================
def load_models(updated_yaml):
    global config, old_config
    gr.Warning(f"Updating config")
    config = OmegaConf.create(updated_yaml)
    reload_pipe = config["pipe_kwargs"]["pipe_id"] != old_config["pipe_kwargs"]["pipe_id"]
    reload_vlm = config["vlm_kwargs"]["vlm_id"] != old_config["vlm_kwargs"]["vlm_id"]
    if reload_pipe:
        global pipe
        del pipe
        gr.Warning(f"Updating pipe")
        pipe = dig_helpers.load_pipe(**config["pipe_kwargs"])
    if reload_vlm:
        global vlm, vlm_processor
        del vlm, vlm_processor
        gr.Warning(f"Updating vlm")
        vlm, vlm_processor = dig_helpers.load_vlm(**config["vlm_kwargs"])
    old_config = config
    torch.cuda.empty_cache()
    return enable_element(2)

def load_config(pipe_name, vlm_name):
    global config
    config = compile_config(pipe_name, vlm_name)
    return OmegaConf.to_yaml(config)

def get_config():
    global config
    return OmegaConf.to_yaml(config)

def disable_element(n):
    return [gr.update(interactive=False)] * n

def enable_element(n):
    return [gr.update(interactive=True)] * n

def preprocess_ref_image(ref_image):
    if ref_image is not None:
        ref_image = ref_image["composite"]
        ref_image = ref_image.convert("RGBA")
        if np.array(ref_image).sum() == 0:
            return None
        background = Image.new("RGBA", ref_image.size, (255, 255, 255))
        ref_image = Image.alpha_composite(background, ref_image)
        ref_image = ref_image.convert("RGB")
        ref_image = ref_image.resize((config["generator_kwargs"]["width"], config["generator_kwargs"]["height"]))
    return ref_image

# ====================
#      Auto Q&A
# ====================
with open("configs/app/gpt_question.txt", "r", encoding="utf-8") as file:
    QA_GPT_PROMPT = file.read()

with open("configs/app/gpt_expand.txt", "r", encoding="utf-8") as file:
    EXPAND_GPT_PROMPT = file.read()

def query_gpt(prompt, gpt_prompt, ref_image, qa_pairs, use_image=True):
    gpt_prompt = gpt_prompt.replace("<prompt>", prompt)
    gpt_prompt = gpt_prompt.replace("<question>", qa_pairs[0][0] if qa_pairs else "None")
    if use_image:
        ref_image = preprocess_ref_image(ref_image)
        ref_image = "None" if ref_image is None else ref_image
        messages = gpt_helpers.create_message(gpt_prompt, "<reference>", [ref_image])
        response = []
    else:
        messages = gpt_helpers.create_message(gpt_prompt)
        response = None
    try:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=messages,
            temperature=1.0
        )
        response = response.choices[0].message.content
        response = json.loads(response.replace("json", "").replace("```", ""))
    except Exception as e:
        gr.Warning("Error parsing GPT-4o output. Try again.")
    return response

with gr.Blocks() as demo:
    gr.Markdown("## Image Generator with Q&A Input")
    
    with gr.Row():
        prompt = gr.Textbox(label="Image Generator Prompt")
    
    expand_prompt_button = gr.Button("🚀 Expand Prompt", interactive=(client is not None))
    feeling_lucky_button = gr.Button("🎲 I'm Feeling Lucky", interactive=(client is not None))
    with gr.Row():
        qa_pairs = gr.Dataframe(headers=["Question", "Answer"], type="array", row_count=(5, "dynamic"))
    
    vanilla_button = gr.Button("Vanilla Generate w/o Optimize", variant="secondary")
    optimize_button = gr.Button("Optimize", variant="primary")
    
    with gr.Row():
        output_image = gr.Image(type="pil", label="Generated Image")
    stop_button = gr.Button("Stop", variant="stop")

    with gr.Accordion("🔧 Config Editor", open=False):
        gr.Markdown(open("configs/app/README_config.md", "r", encoding="utf-8").read())
        pipe_name = gr.Dropdown(choices=PIPE_CONFIGS, value=PIPE_DEFAULT, label="Image Generator")
        vlm_name = gr.Dropdown(choices=VLM_CONFIGS, value=VLM_DEFAULT, label="VLM")
        config_editor = gr.Code(value=get_config, language="yaml", label="Edit Config")
        models_button = gr.Button("Update Config")

    with gr.Accordion("🎲 GPT Q&A Editor", open=False):
        qa_prompt = gr.Textbox(lines=4, value=QA_GPT_PROMPT)
    
    with gr.Accordion("🚀 GPT Expand Editor", open=False):
        expand_prompt = gr.Textbox(lines=4, value=EXPAND_GPT_PROMPT)

    with gr.Accordion("🎨 VLM Prompt Editor", open=False):
        overlay_mode = gr.Radio(choices=["transparent", "solid"], value="transparent", label="Overlay Mode")
        ref_image = gr.ImageEditor(type="pil", label="Upload Ref Image", width=config["generator_kwargs"]["width"], height=config["generator_kwargs"]["height"])
        clear_ref_image_button = gr.Button("Clear Ref Image")

    # Create interactions
    expand_prompt_button.click(lambda a, b: query_gpt(a, b, None, None, False)["prompt"], inputs=[prompt, expand_prompt], outputs=[prompt])
    feeling_lucky_button.click(lambda a, b, c, d: query_gpt(a, b, c, d, True), inputs=[prompt, qa_prompt, ref_image, qa_pairs], outputs=[qa_pairs])
    vanilla_button.click(vanilla, inputs=[prompt, qa_pairs, ref_image, overlay_mode], outputs=[output_image])
    optimize_button.click(optimize, inputs=[prompt, qa_pairs, ref_image, overlay_mode], outputs=[output_image])
    stop_button.click(clear_page, outputs=[output_image])
    pipe_name.change(load_config, inputs=[pipe_name, vlm_name], outputs=[config_editor])
    vlm_name.change(load_config, inputs=[pipe_name, vlm_name], outputs=[config_editor])
    models_button.click(lambda: disable_element(2), outputs=[vanilla_button, optimize_button])
    models_button.click(load_models, inputs=[config_editor], outputs=[vanilla_button, optimize_button])
    clear_ref_image_button.click(lambda: None, outputs=[ref_image])

demo.launch(share=True)