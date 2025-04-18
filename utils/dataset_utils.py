import torch

class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy
    


class CutMix_AUG:
    def __init__(self, alpha=1.0):
        """
        CutMix Augmentation class
        :param alpha: The parameter for the Beta distribution to control the proportion of mixing
        """
        self.alpha = alpha
        self.dist = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def aug(self, rgb_gt, rgb_noisy):
        """
        Apply CutMix augmentation to the images.
        
        :param rgb_gt: Ground truth clean images, shape (batch_size, channels, height, width)
        :param rgb_noisy: Noisy images, shape (batch_size, channels, height, width)
        :return: Mixed ground truth and noisy images
        """
        bs = rgb_gt.size(0)  # Batch size

        # Randomly permute indices for mixing
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        # Sample lambda from Beta distribution
        lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1).cuda()  # Sampling lambda and reshaping for broadcasting

        # Cut and paste: Create a random bounding box for the patch
        height, width = rgb_gt.size(2), rgb_gt.size(3)
        bbx1 = torch.randint(0, height, (bs, 1)).cuda()
        bby1 = torch.randint(0, width, (bs, 1)).cuda()
        bbx2 = torch.randint(bbx1, height, (bs, 1)).cuda()
        bby2 = torch.randint(bby1, width, (bs, 1)).cuda()

        # Apply CutMix: cut one image and paste it on the other
        for i in range(bs):
            rgb_gt[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = rgb_gt2[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]]
            rgb_noisy[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = rgb_noisy2[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]]

        return rgb_gt, rgb_noisy

