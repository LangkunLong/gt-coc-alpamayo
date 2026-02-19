import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1, AlpamayoR1Config
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


CLIP_LIST_PATH = "notebooks/clip_ids.parquet"  # around 1k clips
OUTPUT_FILE = "physical_ai_inference_results.csv"
MODEL_PATH = "nvidia/Alpamayo-R1-10B"
NUM_SAMPLES = ""

def main():

    print(f"Loading clip list from {CLIP_LIST_PATH}...")
    try:
        df_clips = pd.read_parquet(CLIP_LIST_PATH)
        clip_ids = df_clips['clip_id'].tolist()
        print(f"Found {len(clip_ids)} total clips available.")
    except FileNotFoundError:
        print(f"Error: Could not find {CLIP_LIST_PATH}.")
        print("Please ensure you are running this from the repo root.")
        return

   
    if NUM_SAMPLES:
        clip_ids = clip_ids[:NUM_SAMPLES]
        print(f"Subsampled to first {NUM_SAMPLES} clips for testing.")

    config = AlpamayoR1Config.from_pretrained("nvidia/Alpamayo-R1-10B")
    config.attn_implementation = "sdpa"
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16, config=config).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    
    results = []
    
    print("Starting Inference Loop...")
    for clip_id in tqdm(clip_ids):
        try:
            
            data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
            messages = helper.create_message(data["image_frames"].flatten(0, 1))
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            }

            model_inputs = helper.to_device(model_inputs, "cuda")

            torch.cuda.manual_seed_all(42)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
                    max_generation_length=256,
                    return_extra=True,
                )

            trace = extra["cot"][0]
            gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
            pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
            diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
            min_ade = diff.min()

            results.append({
                "clip_id": clip_id,
                "reasoning_trace": trace,
                "min_ade": min_ade,
                "status": "Success"
            })

            if len(results) % 5 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

        except Exception as e:
            print(f"\nFailed on {clip_id}: {e}")
            results.append({
                "clip_id": clip_id,
                "reasoning_trace": "",
                "min_ade": "",
                "status": f"Error: {str(e)}"
            })


    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Processed {len(results)} clips. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()