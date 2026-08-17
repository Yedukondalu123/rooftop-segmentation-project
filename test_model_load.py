import torch
from building_footprint_segmentation.seg.binary.models import ReFineNet

def get_model(weight_path):
    model = ReFineNet()
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

if __name__ == "__main__":
    model = get_model("examples/refine.pth")
    print("Model loaded successfully")

    # dummy input: batch=1, channels=3, height=256, width=256
    dummy = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy)
        output = output.sigmoid()

    print("Output shape:", output.shape)
    print("Output min/max:", output.min().item(), output.max().item())