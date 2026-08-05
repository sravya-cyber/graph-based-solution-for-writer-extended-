import os
import re
import time
import torch
import numpy as np
from multiprocessing import Pool, cpu_count

from preprocessing import imread_unicode, get_clean_ink_skeleton
from kp_detection import detect_keypoints
from graph_construction import build_simple_paper_graph

DATASET_ROOT = "/kaggle/input/datasets/sravyakalusani/cvl-dataset-clean"
CACHE_FILE = "cvl_graph_cache.pt"
MAX_NODES = 50

def extract_classical_features(img, skel, norm_pos):
    """Extracts 11 hand-crafted word-level classical features."""
    feats = np.zeros(11, dtype=np.float32)
    if img is None or skel is None:
        return feats

    h, w = img.shape[:2]
    ink_pts = np.argwhere(skel > 0)
    total_ink = len(ink_pts)

    # 1. Word Aspect Ratio
    feats[0] = w / float(h + 1e-6)
    # 2. Ink Density
    feats[1] = total_ink / float(h * w + 1e-6)
    
    if total_ink > 5:
        # 3. Horizontal & Vertical Ink Center Spread
        y_std, x_std = ink_pts.std(axis=0)
        feats[2] = x_std / float(w + 1e-6)
        feats[3] = y_std / float(h + 1e-6)
        
        # 4. Global Slant / Orientation (Principal Axis)
        coords = ink_pts - ink_pts.mean(axis=0)
        cov = np.cov(coords.T)
        eigvals, _ = np.linalg.eig(cov)
        feats[4] = eigvals.max() / float(eigvals.min() + 1e-6)

    if len(norm_pos) > 1:
        # 5. Average Node Distance (Spacing)
        dists = [np.linalg.norm(norm_pos[i] - norm_pos[j]) 
                 for i in range(len(norm_pos)) for j in range(i+1, len(norm_pos))]
        feats[5] = np.mean(dists) if dists else 0.0
        feats[6] = np.std(dists) if dists else 0.0

    # 7-10. Bounding Box Norms & Node Density
    feats[7] = len(norm_pos) / 50.0  # Keypoint ratio
    feats[8] = norm_pos[:, 0].std() if len(norm_pos) > 1 else 0.0
    feats[9] = norm_pos[:, 1].std() if len(norm_pos) > 1 else 0.0
    feats[10] = total_ink / float(len(norm_pos) + 1e-6) # Ink per keypoint

    return feats

def process_image(sample):
    img = imread_unicode(sample["path"])
    node_feats = np.zeros((MAX_NODES, 7), dtype=np.float32)
    adj_matrix = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    mask = np.zeros(MAX_NODES, dtype=np.float32)
    classical_feats = np.zeros(11, dtype=np.float32)

    if img is not None:
        _, skel = get_clean_ink_skeleton(img)
        if skel is not None:
            kps = detect_keypoints(skel, dist_thresh_px=3.0, angle_thresh_deg=25.0)
            G, _, _ = build_simple_paper_graph(skel, kps, merge_dist_px=4.0)

            if G is not None and G.number_of_nodes() > 0:
                raw_node_count = G.number_of_nodes()
                num_nodes = min(raw_node_count, MAX_NODES)
                
                positions = np.array([G.nodes[i]["pos_norm"] for i in range(num_nodes)], dtype=np.float32)
                min_pos, max_pos = positions.min(axis=0), positions.max(axis=0)
                span = np.where((max_pos - min_pos) == 0, 1.0, (max_pos - min_pos))
                norm_pos = (positions - min_pos) / span
                center = norm_pos.mean(axis=0)

                for i in range(num_nodes):
                    x_n, y_n = norm_pos[i]
                    deg = G.degree[i] if i in G else 0
                    node_feats[i, 0] = x_n
                    node_feats[i, 1] = y_n
                    node_feats[i, 2] = x_n - center[0]
                    node_feats[i, 3] = y_n - center[1]
                    node_feats[i, 4] = deg / 10.0
                    node_feats[i, 5] = 1.0 if deg == 1 else 0.0
                    node_feats[i, 6] = 1.0 if deg >= 3 else 0.0
                    mask[i] = 1.0

                for u in range(num_nodes):
                    for v in range(num_nodes):
                        if u != v and G.has_edge(u, v):
                            dist = np.linalg.norm(norm_pos[u] - norm_pos[v])
                            adj_matrix[u, v] = np.exp(-dist * 2.0)

                classical_feats = extract_classical_features(img, skel, norm_pos)

                return {
                    "x": node_feats,
                    "adj": adj_matrix,
                    "mask": mask,
                    "classical": classical_feats,
                    "label": sample["label"],
                    "page_num": sample["page_num"]
                }
    return None

def main():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("Removed old cache file.")

    print(f"Scanning dataset directory: {DATASET_ROOT} ...")
    all_image_paths = []
    for root, _, files in os.walk(DATASET_ROOT):
        if "words" in root.lower():
            for f in files:
                if f.lower().endswith((".tif", ".png", ".jpg", ".tiff")):
                    all_image_paths.append(os.path.join(root, f))

    writer_to_label = {}
    samples = []
    for path in all_image_paths:
        filename = os.path.basename(path)
        numbers = re.findall(r"\d+", filename)
        if len(numbers) >= 2:
            writer_id = numbers[0].zfill(4)
            page_num = int(numbers[1])
            if page_num in [1, 2, 3, 4, 6]:
                if writer_id not in writer_to_label:
                    writer_to_label[writer_id] = len(writer_to_label)
                samples.append({"path": path, "label": writer_to_label[writer_id], "page_num": page_num})

    total_samples = len(samples)
    print(f"Total word crops found: {total_samples}")

    num_workers = max(1, cpu_count() - 1)
    start_time = time.time()
    cached_data = []

    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap(process_image, samples, chunksize=20), start=1):
            if result is not None:
                cached_data.append({
                    "x": torch.from_numpy(result["x"]),
                    "adj": torch.from_numpy(result["adj"]),
                    "mask": torch.from_numpy(result["mask"]),
                    "classical": torch.from_numpy(result["classical"]),
                    "label": result["label"],
                    "page_num": result["page_num"]
                })
            
            if i % 100 == 0 or i == total_samples:
                elapsed = time.time() - start_time
                pct = (i / total_samples) * 100
                print(f"Extracting Hybrid Cache: [{i:5d}/{total_samples}] ({pct:5.1f}%) | Elapsed: {elapsed:.1f}s")

    print("\nSaving hybrid graph cache to disk...")
    torch.save(cached_data, CACHE_FILE)
    print(f"Done! Saved {len(cached_data)} valid word graphs to '{CACHE_FILE}'.")

if __name__ == "__main__":
    main()
