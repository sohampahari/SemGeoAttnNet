#!/usr/bin/env python3
import argparse
import os
import glob
from time import time
from datetime import datetime

import torch

from sem_ft import get_features_per_vertex
from diffusion import init_pipe
from dino import init_dino
from mesh_container import MeshContainer
from utils import convert_mesh_container_to_torch_mesh

def parse_args():
    p = argparse.ArgumentParser(description="Extract per-vertex features from a folder of meshes using diff3f.")
    p.add_argument("--mesh_glob", default="meshes/*.obj", help="glob for your mesh files (default: meshes/*.obj)")
    p.add_argument("--out_dir", default=None, help="where to save .pt features (default: output/meshes/<timestamp>)")
    p.add_argument("--num_views", type=int, default=100, help="number of rendered views per mesh")
    p.add_argument("--H", type=int, default=512, help="render height")
    p.add_argument("--W", type=int, default=512, help="render width")
    p.add_argument("--tolerance", type=float, default=0.01, help="tolerance used when aggregating features")
    p.add_argument("--num_images_per_prompt", type=int, default=1)
    p.add_argument("--use_normal_map", action="store_true", help="use normal map conditioning")
    p.add_argument("--prompt_mode", choices=["filename", "custom", "null"], default="filename",
                   help="how to derive the text prompt: from filename (letters-only), custom (same for all), or null")
    p.add_argument("--prompt", default="object", help="custom prompt when --prompt_mode custom is selected")
    p.add_argument("--device", default=None, help="torch device (e.g. cuda:0). If omitted, auto-detect.")
    return p.parse_args()

def make_prompt_from_filename(filename):
    # keep only letters (same heuristic as extract_tosca.py)
    return "".join(filter(str.isalpha, filename))

def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now = datetime.now().strftime("%d-%m-%Y_%H-%M")
    out_dir = args.out_dir or f"output/meshes/{now}"
    os.makedirs(out_dir, exist_ok=True)

    target_files = sorted(glob.glob(args.mesh_glob))
    if not target_files:
        print(f"No meshes found for glob: {args.mesh_glob}")
        return

    print(f"Found {len(target_files)} meshes. Saving features to {out_dir}")
    print("Initializing models on device:", device)
    t0 = time()
    pipe = init_pipe(device, use_normal_map=args.use_normal_map)
    dino_model = init_dino(device)

    processed = 0
    for f in target_files:
        try:
            print(f"\nProcessing: {f}")
            filename = os.path.basename(f).split(".")[0]
            if args.prompt_mode == "filename":
                prompt = make_prompt_from_filename(filename) or args.prompt
            elif args.prompt_mode == "custom":
                prompt = args.prompt
            else:  # null
                prompt = ""

            print("Prompt:", repr(prompt))
            mesh_container = MeshContainer().load_from_file(str(f))
            mesh = convert_mesh_container_to_torch_mesh(mesh_container, device=device, is_tosca=False)
            mesh_vertices = mesh.verts_list()[0]

            features = get_features_per_vertex(
                device=device,
                pipe=pipe,
                dino_model=dino_model,
                mesh=mesh,
                prompt=prompt,
                mesh_vertices=mesh_vertices,
                num_views=args.num_views,
                H=args.H,
                W=args.W,
                tolerance=args.tolerance,
                num_images_per_prompt=args.num_images_per_prompt,
                use_normal_map=args.use_normal_map,
            )

            save_name = os.path.join(out_dir, f"{filename}.pt")
            torch.save(features, save_name)
            print(f"Saved features to {save_name} (tensor shape: {getattr(features, 'shape', 'unknown')})")
            processed += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print(f"\nDone. Processed {processed}/{len(target_files)} meshes in {(time() - t0)/60:.2f} minutes.")
    print(f"Feature files are in: {out_dir}")

if __name__ == "__main__":
    main()