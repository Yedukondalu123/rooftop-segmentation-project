import torch

ckpt = torch.load("examples/refine.pth", map_location="cpu")
print("Type:", type(ckpt))

if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print("Number of keys:", len(keys))
    print("First 10 keys:", keys[:10])
    print("Last 5 keys:", keys[-5:])

    # check if any of the keys are actually a nested wrapper like 'model' or 'state_dict'
    for k, v in list(ckpt.items())[:5]:
        if hasattr(v, "shape"):
            print(f"{k}: tensor shape {tuple(v.shape)}")
        else:
            print(f"{k}: type {type(v)}")