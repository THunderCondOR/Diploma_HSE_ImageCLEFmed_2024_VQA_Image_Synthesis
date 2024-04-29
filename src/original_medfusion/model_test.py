from medical_diffusion.models.embedders.latent_embedders import VAE, VAELoss
from dataset import ClefMedfusionVAEDataset


vae = VAE(
        in_channels=3, 
        out_channels=3, 
        emb_channels=8,
        spatial_dims=2,
        hid_chs =[64, 128, 256,  512], 
        kernel_sizes=[3,  3,   3,    3],
        strides=[1,  2,   2,    2],
        deep_supervision=1,
        use_attention='none',
    )
