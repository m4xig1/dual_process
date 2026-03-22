from omegaconf import OmegaConf
import os
import sys

from dual_process import dig_helpers, dig_pipeline

def init_config():
    global_config = OmegaConf.create({})
    for config_name in sys.argv[1].split(","):
        global_config = OmegaConf.merge(global_config, OmegaConf.load(config_name))
    if len(sys.argv) > 2:
        cli_overrides = OmegaConf.from_cli(sys.argv[2:])
        global_config = OmegaConf.merge(global_config, cli_overrides)
    OmegaConf.resolve(global_config)
    save_folder = global_config["save_folder"]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok="True")
    OmegaConf.save(global_config, f"{save_folder}/config.yaml")
    return global_config, save_folder

if __name__ == "__main__":
    config, save_folder = init_config()
    pipe = dig_helpers.load_pipe(**config["pipe_kwargs"])
    vlm, vlm_processor = dig_helpers.load_vlm(**config["vlm_kwargs"])

    for sample in config["dataset"]:
        prompt = sample["prompt"]
        qa_pairs = sample["qa_pairs"]
        name = sample["name"]
        opt_repeat = config.get("opt_repeat", 1)
        opt_start = config.get("opt_start", 0)
        for n_repeat in range(opt_repeat):
            qa_folder = f"{save_folder}/{name}/{n_repeat + opt_start}"
            # Create save folder
            if os.path.exists(qa_folder) and len(os.listdir(qa_folder)) > 0:
                print(f"Skipping {name} because sample exists and folder is non-empty.")
                continue
            os.makedirs(qa_folder, exist_ok="True")
            # Create edit and params
            edit = dig_pipeline.create_edit(pipe, vlm, vlm_processor, config, qa_pairs, prompt)
            params = dig_helpers.create_lora(pipe, **config["lora_kwargs"])
            dig_pipeline.optimize(pipe, edit, config["opt_kwargs"], config["generator_kwargs"], [params], prompt, qa_folder, config)