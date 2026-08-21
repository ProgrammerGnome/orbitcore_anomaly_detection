import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from anomalib.models.components import KCenterGreedy, GaussianBlur2d
from torchvision.transforms import v2

def create_homography_grid(p, B, H_out, W_out, device):
    y, x = torch.meshgrid(torch.linspace(-1, 1, H_out, device=device),
                          torch.linspace(-1, 1, W_out, device=device), indexing='ij')
    ones = torch.ones_like(x)
    grid = torch.stack([x, y, ones], dim=0).view(3, -1) 
    
    r1 = torch.stack([1.0 + p[:, 0], p[:, 1], p[:, 2]], dim=1)
    r2 = torch.stack([p[:, 3], 1.0 + p[:, 4], p[:, 5]], dim=1)
    r3 = torch.stack([p[:, 6], p[:, 7], torch.ones(B, dtype=p.dtype, device=device)], dim=1)
    H_mat = torch.stack([r1, r2, r3], dim=1) 
    
    grid_T = torch.bmm(H_mat, grid.unsqueeze(0).expand(B, -1, -1))
    
    x_T = grid_T[:, 0, :] / (grid_T[:, 2, :] + 1e-6)
    y_T = grid_T[:, 1, :] / (grid_T[:, 2, :] + 1e-6)
    
    x_T = x_T.view(B, H_out, W_out)
    y_T = y_T.view(B, H_out, W_out)
    
    return torch.stack([x_T, y_T], dim=-1)

def create_inverse_homography_grid(p, B, H_out, W_out, device):
    y, x = torch.meshgrid(torch.linspace(-1, 1, H_out, device=device),
                          torch.linspace(-1, 1, W_out, device=device), indexing='ij')
    ones = torch.ones_like(x)
    grid = torch.stack([x, y, ones], dim=0).view(3, -1) 
    
    r1 = torch.stack([1.0 + p[:, 0], p[:, 1], p[:, 2]], dim=1)
    r2 = torch.stack([p[:, 3], 1.0 + p[:, 4], p[:, 5]], dim=1)
    r3 = torch.stack([p[:, 6], p[:, 7], torch.ones(B, dtype=p.dtype, device=device)], dim=1)
    H_mat = torch.stack([r1, r2, r3], dim=1) 
    
    H_inv = torch.linalg.inv(H_mat)
    
    grid_T = torch.bmm(H_inv, grid.unsqueeze(0).expand(B, -1, -1))
    
    x_T = grid_T[:, 0, :] / (grid_T[:, 2, :] + 1e-6)
    y_T = grid_T[:, 1, :] / (grid_T[:, 2, :] + 1e-6)
    
    x_T = x_T.view(B, H_out, W_out)
    y_T = y_T.view(B, H_out, W_out)
    
    return torch.stack([x_T, y_T], dim=-1)

class TTAHomography(nn.Module):
    def __init__(self, max_iter=10, lr=0.05, pool_size=(8, 8)):
        super().__init__()
        self.max_iter = max_iter
        self.lr = lr
        self.pool_size = pool_size
        self.last_p = None

    def forward(self, f_test, f_ref):
        B, C, H, W = f_test.shape

        with torch.inference_mode(False):
            with torch.enable_grad():
                f_test_var = f_test.clone().detach()
                f_ref_var = f_ref.clone().detach()
                
                p = torch.zeros(B, 8, device=f_test.device, dtype=f_test.dtype, requires_grad=True)
                optimizer = torch.optim.Adam([p], lr=self.lr)
                
                f_ref_pool = F.adaptive_avg_pool2d(f_ref_var.expand(B, -1, -1, -1), self.pool_size)
                f_ref_flat = F.normalize(f_ref_pool.view(B, C, -1), p=2, dim=2)

                for _ in range(self.max_iter):
                    optimizer.zero_grad()
                    grid = create_homography_grid(p, B, H, W, f_test_var.device)
                    f_warp = F.grid_sample(f_test_var, grid, align_corners=False, padding_mode='border')
                    
                    f_warp_pool = F.adaptive_avg_pool2d(f_warp, self.pool_size)
                    f_warp_flat = F.normalize(f_warp_pool.view(B, C, -1), p=2, dim=2)
                    
                    ecc = (f_warp_flat * f_ref_flat).sum(dim=2).mean(dim=1)
                    loss = (1.0 - ecc).mean()
                    
                    loss.backward()
                    optimizer.step()

        with torch.no_grad():
            self.last_p = p.detach() 
            grid = create_homography_grid(p.detach(), B, H, W, f_test.device)
            f_final = F.grid_sample(f_test, grid, align_corners=False, padding_mode='border')
            
        return f_final

class OrbitCoreModel(nn.Module):
    def __init__(self, layers=["layer2", "layer3"], target_dim=128, kernel_size=3, coreset_sampling_ratio=0.01, use_srp=True, orbit_alpha=0.2):
        super().__init__()
        self.backbone = resnet18(weights="IMAGENET1K_V1")
        self.layers = layers
        
        resnet18_dims = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}

        self.feature_dim = sum([resnet18_dims[l] for l in self.layers])

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.use_srp = use_srp
        if self.use_srp:
            self.register_buffer("srp_matrix", torch.randn(target_dim, self.feature_dim))
            
        self.tta_module = TTAHomography(max_iter=30, lr=0.075, pool_size=(8, 8)) 
        self.kernel_size = kernel_size
        self.coreset_sampling_ratio = coreset_sampling_ratio
        
        self.orbit_alpha = orbit_alpha
        
        self.reference_layer1 = None 
        self.memory_bank = None
        self.register_buffer("noise_floor", torch.tensor(0.0)) 
        
        sigma = 4
        blur_kernel_size = 2 * int(4.0 * sigma + 0.5) + 1 
        self.anomaly_map_generator = GaussianBlur2d(
            kernel_size=(blur_kernel_size, blur_kernel_size), 
            sigma=(sigma, sigma), 
            channels=1
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval() 
        return self

    def extract_features(self, x):
        pass

    def forward(self, x):
        B = x.shape[0]
        
        with torch.no_grad():
            x_feat = self.backbone.conv1(x)
            x_feat = self.backbone.bn1(x_feat)
            x_feat = self.backbone.relu(x_feat)
            x_feat = self.backbone.maxpool(x_feat)
            f1 = self.backbone.layer1(x_feat) 
            
        if self.training:
            if self.reference_layer1 is None:
                self.reference_layer1 = f1[0:1].detach().clone()
        else:
            ref_layer = self.reference_layer1.to(f1.device)
            f1 = self.tta_module(f1, ref_layer)
            
        features = {}
        if "layer1" in self.layers:
            features["layer1"] = f1
            
        with torch.no_grad():
            f2 = self.backbone.layer2(f1) 
            features["layer2"] = f2
            f3 = self.backbone.layer3(f2) 
            features["layer3"] = f3
            if "layer4" in self.layers:
                f4 = self.backbone.layer4(f3)
                features["layer4"] = f4
            
        extracted = [features[l] for l in self.layers]

        target_size = extracted[0].shape[2:]
        resized_features = []
        for f in extracted:
            if f.shape[2:] != target_size:
                resized_features.append(F.interpolate(f, size=target_size, mode='bilinear', align_corners=False))
            else:
                resized_features.append(f)
        f_raw = torch.cat(resized_features, dim=1)

        if self.use_srp:
            f_srp = torch.einsum('dc, bchw -> bdhw', self.srp_matrix.to(f_raw.device), f_raw)
        else:
            f_srp = f_raw

        f_srp = F.normalize(f_srp, p=2, dim=1)
        
        pad = self.kernel_size // 2
        mu = F.avg_pool2d(f_srp, self.kernel_size, stride=1, padding=pad)
        f_srp_sq = F.avg_pool2d(f_srp**2, self.kernel_size, stride=1, padding=pad)
        sigma = torch.sqrt(torch.clamp(f_srp_sq - mu**2, min=1e-5))
        phi = torch.cat([mu, sigma], dim=1) 
        
        H_phi, W_phi = phi.shape[2], phi.shape[3]
        a = self.orbit_alpha 
        
        orbit_params = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -a, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -a]
        ], device=phi.device, dtype=phi.dtype)
        
        perspective_phis = []
        for p_orbit in orbit_params:
            p_batch = p_orbit.unsqueeze(0).expand(B, -1)
            grid = create_homography_grid(p_batch, B, H_phi, W_phi, phi.device)
            phi_warped = F.grid_sample(phi, grid, align_corners=False, padding_mode='border')
            perspective_phis.append(phi_warped)
            
        stacked_phis = torch.stack(perspective_phis, dim=0) 

        orbit_mu = stacked_phis.mean(dim=0)
        orbit_s = stacked_phis.std(dim=0, unbiased=False)
        z = torch.cat([orbit_mu, orbit_s], dim=1) 
        
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        
        if self.training:
            return z_flat
        else:
            mb = self.memory_bank.to(z_flat.device)
            distances = torch.cdist(z_flat, mb, p=2.0)
            min_distances, _ = torch.min(distances, dim=1)
            
            anomaly_map_raw = min_distances.reshape(B, z.shape[2], z.shape[3])

            mask = torch.ones_like(anomaly_map_raw)
            margin = 2 
            mask[:, :margin, :] = 0.0  
            mask[:, -margin:, :] = 0.0 
            mask[:, :, :margin] = 0.0  
            mask[:, :, -margin:] = 0.0 
            anomaly_map_raw = anomaly_map_raw * mask
            
            flat_maps = anomaly_map_raw.reshape(B, -1)
            k_val = max(1, int(flat_maps.shape[1] * 0.02)) 
            topk_vals, _ = torch.topk(flat_maps, k=k_val, dim=1)
            anomaly_score = topk_vals.mean(dim=1)
            
            anomaly_map = F.interpolate(
                anomaly_map_raw.unsqueeze(1), 
                size=(x.shape[2], x.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
            
            anomaly_map = self.anomaly_map_generator(anomaly_map)
            
            p = self.tta_module.last_p
            inv_grid = create_inverse_homography_grid(p, B, anomaly_map.shape[2], anomaly_map.shape[3], anomaly_map.device)
            anomaly_map = F.grid_sample(anomaly_map, inv_grid, align_corners=False, padding_mode='zeros')
            
            return anomaly_score, anomaly_map
            
    def fit_coreset(self, features):
        sampler = KCenterGreedy(embedding=features, sampling_ratio=self.coreset_sampling_ratio)
        self.memory_bank = sampler.sample_coreset()

        with torch.no_grad():
            subset_size = min(10000, features.shape[0])
            subset_idx = torch.randperm(features.shape[0])[:subset_size]
            
            device = next(self.backbone.parameters()).device
            sub_features_gpu = features[subset_idx].to(device)
            mb_gpu = self.memory_bank.to(device)
            
            dists = torch.cdist(sub_features_gpu, mb_gpu)
            min_dists, _ = torch.min(dists, dim=1)

            self.noise_floor = torch.quantile(min_dists, 0.99).cpu()
            
            del sub_features_gpu, mb_gpu, dists, min_dists
            torch.cuda.empty_cache()
