import sys, time, torch, timm
torch.backends.cudnn.benchmark = True
arch, size, bs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = timm.create_model(arch, pretrained=True, num_classes=11, in_chans=1).cuda().to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(m.parameters(), 1e-4)
x = torch.randn(bs, 1, size, size, device="cuda").contiguous(memory_format=torch.channels_last)
for i in range(13):
    if i == 3: torch.cuda.synchronize(); t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l = m(x).float().mean()
    opt.zero_grad(); l.backward(); opt.step()
torch.cuda.synchronize()
ips = 10 * bs / (time.time() - t)
print(f"{arch} {size} bs{bs}: {ips:.0f} img/s, epoch(21.6k)={21660/ips/60:.1f} min, mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
