import numpy as np
import networkx as nx
from skan import Skeleton

def build_simple_paper_graph(skeleton_img, kps, merge_dist_px=4.0):
    h, w = skeleton_img.shape
    bool_skel = (skeleton_img > 0) if np.max(skeleton_img) > 1 else (skeleton_img == 1)
    
    if bool_skel.sum() < 3 or not kps:
        return nx.Graph(), w, h

    try:
        skel_obj = Skeleton(bool_skel)
    except Exception:
        return nx.Graph(), w, h

    kp_nodes = []
    for idx, (kx, ky) in enumerate(kps):
        x_norm = round((kx / max(w, 1)) * 10.0, 2)
        y_norm = round(10.0 - ((ky / max(h, 1)) * 10.0), 2)
        kp_nodes.append({
            "id": idx,
            "pixel": (int(kx), int(ky)),
            "norm": (x_norm, y_norm),
        })

    adj = {i: [] for i in range(len(kp_nodes))}

    for path_idx in range(skel_obj.n_paths):
        coords = skel_obj.path_coordinates(path_idx)
        stroke = [(int(c[1]), int(c[0])) for c in coords]

        if len(stroke) < 2:
            continue

        path_keypoints = []
        for kp_idx, node in enumerate(kp_nodes):
            kx, ky = node["pixel"]
            dists = [np.hypot(pt[0] - kx, pt[1] - ky) for pt in stroke]
            if min(dists) <= merge_dist_px:
                best_pos = int(np.argmin(dists))
                path_keypoints.append((best_pos, kp_idx))

        if not path_keypoints:
            continue

        path_keypoints.sort(key=lambda item: item[0])

        for k in range(len(path_keypoints) - 1):
            u_idx = path_keypoints[k][1]
            v_idx = path_keypoints[k + 1][1]
            if u_idx != v_idx and v_idx not in adj[u_idx]:
                adj[u_idx].append(v_idx)
            if v_idx != u_idx and u_idx not in adj[v_idx]:
                adj[v_idx].append(u_idx)

    G = nx.Graph()
    for node_idx, node in enumerate(kp_nodes):
        G.add_node(node_idx, pos_px=node["pixel"], pos_norm=node["norm"])

    for u_idx in range(len(kp_nodes)):
        for v_idx in adj[u_idx]:
            if u_idx >= v_idx:
                continue
            if not G.has_edge(u_idx, v_idx):
                x1, y1 = G.nodes[u_idx]["pos_norm"]
                x2, y2 = G.nodes[v_idx]["pos_norm"]
                w_ij = round(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)
                G.add_edge(u_idx, v_idx, weight=w_ij)

    return G, w, h
