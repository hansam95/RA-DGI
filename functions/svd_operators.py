"""SVD operators retained from the original DGI implementation; see THIRD_PARTY.md."""

import torch


class A_functions:
    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def A(self, vec):
        """
        Multiplies the input vector by A
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, : singulars.shape[0]])

    def At(self, vec):
        """
        Multiplies the input vector by A transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def A_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of A
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = 1.0 / singulars
        factors[singulars == 0] = 0.0
        temp[:, : singulars.shape[0]] = temp[:, : singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))


class CS(A_functions):
    def __init__(self, channels, img_dim, ratio, device):
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // 32
        self.ratio = 32
        A = torch.randn(32**2, 32**2).to(device)
        _, _, self.V_small = torch.svd(A, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)
        self.singulars_small = torch.ones(int(32 * 32 * ratio), device=device)
        self.cs_size = self.singulars_small.size(0)

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(
            vec.size(0), self.channels * self.y_dim**2, self.ratio**2, device=vec.device
        )
        patches[:, :, : self.cs_size] = (
            temp[:, : self.channels * self.y_dim**2 * self.cs_size]
            .contiguous()
            .reshape(vec.size(0), -1, self.cs_size)
        )
        patches[:, :, self.cs_size :] = (
            temp[:, self.channels * self.y_dim**2 * self.cs_size :]
            .contiguous()
            .reshape(vec.size(0), self.channels * self.y_dim**2, -1)
        )
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )
        patches_orig = patches.reshape(
            vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio
        )
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim**2)
        return recon

    def Vt(self, vec):
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, : self.channels * self.y_dim**2 * self.cs_size] = (
            patches[:, :, :, : self.cs_size].contiguous().reshape(vec.shape[0], -1)
        )
        recon[:, self.channels * self.y_dim**2 * self.cs_size :] = (
            patches[:, :, :, self.cs_size :].contiguous().reshape(vec.shape[0], -1)
        )
        return recon

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp


class Denoising(A_functions):
    def __init__(self, channels, img_dim, device):
        self._singulars = torch.ones(channels * img_dim**2, device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


class SuperResolution(A_functions):
    def __init__(self, channels, img_dim, ratio, device):
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        A = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(A, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(
            vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device
        )
        patches[:, :, :, 0] = temp[:, : self.channels * self.y_dim**2].view(
            vec.shape[0], self.channels, -1
        )
        for idx in range(self.ratio**2 - 1):
            patches[:, :, :, idx + 1] = temp[
                :, self.channels * self.y_dim**2 + idx :: self.ratio**2 - 1
            ].view(vec.shape[0], self.channels, -1)
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )
        patches_orig = patches.reshape(
            vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio
        )
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim**2)
        return recon

    def Vt(self, vec):
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, : self.channels * self.y_dim**2] = patches[:, :, :, 0].view(
            vec.shape[0], self.channels * self.y_dim**2
        )
        for idx in range(self.ratio**2 - 1):
            recon[:, self.channels * self.y_dim**2 + idx :: self.ratio**2 - 1] = patches[
                :, :, :, idx + 1
            ].view(vec.shape[0], self.channels * self.y_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp
