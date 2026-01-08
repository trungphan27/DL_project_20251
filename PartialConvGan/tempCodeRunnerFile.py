
G = PConvUNet()
G.load_state_dict(torch.load(CKPT, map_location=device))
G = G.to(device)
G.eval()

dataset = Testset(NPY_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

psnr_masked_list = []
psnr_full_list = []
ssim_full_list = []
mse_masked_list = []
mse_unmasked_list = []

fid_model = InceptionFID(device)
real_acts, fake_acts = [], []

with torch.no_grad():
    for batch in tqdm(loader):

        masked = batch[0].to(device)   
        mask   = batch[1].to(device)   
        real   = batch[2].to(device)   

        fake = G(masked, mask)

        composite = fake * mask + real * (1 - mask)

        fake_np = fake.cpu().numpy()
        real_np = real.cpu().numpy()
        comp_np = composite.cpu().numpy()
        mask_np = mask.cpu().numpy()
        inv_mask_np = 1.0 - mask_np

        for i in range(fake_np.shape[0]):

            psnr_masked_list.append(
                peak_signal_noise_ratio(
                    real_np[i] * mask_np[i],
                    fake_np[i] * mask_np[i],
                    data_range=2
                )
            )

            psnr_full_list.append(
                peak_signal_noise_ratio(
                    real_np[i],
                    comp_np[i],
                    data_range=2
                )
            )

            ssim_full_list.append(
                structural_similarity(
                    real_np[i].transpose(1, 2, 0),
                    comp_np[i].transpose(1, 2, 0),
                    channel_axis=2,
                    data_range=2
                )
            )