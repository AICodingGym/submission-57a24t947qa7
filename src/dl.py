import sys, timm
for n in sys.argv[1:]:
    try:
        m = timm.create_model(n, pretrained=True); print('OK', n, sum(p.numel() for p in m.parameters())/1e6, m.pretrained_cfg.get('input_size'), m.pretrained_cfg.get('crop_pct'), m.pretrained_cfg.get('mean'), flush=True); del m
    except Exception as e: print('FAIL', n, e, flush=True)
