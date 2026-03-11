z = torch.randn(1,128).cuda()

fake = G(z)

volume = fake[0,0].cpu().numpy()

save_nii(volume,"generated.nii.gz")