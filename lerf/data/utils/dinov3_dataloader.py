import torch
from typing import Dict
from transformers import AutoImageProcessor, AutoModel
from lerf.data.utils.feature_dataloader import FeatureDataloader
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm        
                    

class Dinov3Dataloader(FeatureDataloader):
    dino_model_type = "dinov3_vitb16"   # 'dinov3_vitb16' | 'dinov3_vitl16' | 'dinov3_vith14'
    dino_stride = 16                    # vitb16=16, vitl16=16, vith14=14
    dino_load_size = 500                
    l2_normalize = False               

    _hf_id_map: Dict[str, str] = {
        "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "dinov3_vith14": "facebook/dinov3-vith14-pretrain-lvd1689m",
    }

    def __init__(self, cfg: dict, device: torch.device, image_list: torch.Tensor, cache_path=None):
        assert "image_shape" in cfg
        super().__init__(cfg, device, image_list, cache_path)
        del image_list

    def _round_to_stride(self, h: int, w: int):
        s = self.dino_stride
        return (h // s) * s, (w // s) * s

    def _prep_hf(self):
        hf_id = self._hf_id_map.get(self.dino_model_type, None)
        if hf_id is None:
            raise ValueError(f"Unknown dino_model_type: {self.dino_model_type}")
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id).to(self.device).eval()

    def create(self, image_list: torch.Tensor):
        self._prep_hf()

        N, _, H0, W0 = image_list.shape

        if H0 <= W0:
            target_h = self.dino_load_size
            target_w = int(round(W0 * (target_h / H0)))
        else:
            target_w = self.dino_load_size
            target_h = int(round(H0 * (target_w / W0)))
        target_h, target_w = self._round_to_stride(target_h, target_w)

        resize_tf = transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.BICUBIC)

        feats = []
        to_pil = transforms.ToPILImage()

        with torch.inference_mode():
            for i in tqdm(range(N), desc="dinov3", total=N, leave=False):
                pil = to_pil(image_list[i].detach().cpu().clamp(0, 1))
                pil = resize_tf(pil)

                inputs = self.processor(images=pil, return_tensors="pt", do_resize=False)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}  

                out = self.model(**inputs)
                tok = out.last_hidden_state  

                Htok = inputs["pixel_values"].shape[-2] // self.dino_stride
                Wtok = inputs["pixel_values"].shape[-1] // self.dino_stride
                expected = Htok * Wtok
                if tok.shape[1] == expected + 1:      
                    tok = tok[:, 1:, :]
                elif tok.shape[1] != expected:
                    raise RuntimeError(
                        f"Token length mismatch: got {tok.shape[1]}, expected {expected} or {expected+1}"
                    )

                fmap = tok.reshape(1, Htok, Wtok, tok.shape[-1]).squeeze(0)
                if self.l2_normalize:
                    fmap = F.normalize(fmap, dim=-1)  

                feats.append(fmap.cpu()) 

        self.data = torch.stack(feats, dim=0)

    def imgpoints_call(self, img_points: torch.Tensor):
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind = (img_points[:, 1] * img_scale[0]).clamp(0, self.data.shape[1]-1).long()
        y_ind = (img_points[:, 2] * img_scale[1]).clamp(0, self.data.shape[2]-1).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)

    def img_call(self, img_ind: int):
        fmap = (self.data[img_ind]).clone().to(self.device)        
        fmap = fmap.permute(2, 0, 1).contiguous()          
        return fmap
