"""Write work/final.json from work/blend_summary.json (+ external model list from the feature dirs used)."""
import os, json, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = json.load(open(f'{ROOT}/work/blend_summary.json'))
start = datetime.datetime(2026, 10, 20, 13, 34)
used = sorted(s['weights'])
models = sorted({n.split('__')[0].rsplit('_', 1)[0] for n in used})
final = {
    'metric': 'multi-class log loss',
    'higher_is_better': False,
    'validation_score': round(s['cv'], 5),
    'validation_method': '5-fold stratified CV on the 9199 train images (out-of-fold predictions); score is the nested CV of the '
                         'blend (blend weights refit on 4/5 of OOF rows, scored on the held-out 1/5). In-sample blend OOF: '
                         f"{s['in_sample']:.5f}",
    'approach': 'Frozen ImageNet-pretrained timm backbones (4-view TTA). Per backbone: ImageNet-1k/22k head logits restricted to the '
                '120 dog synsets + learned temperature + L2-regularised linear residual on pooled features (5-fold CV). '
                'Final = non-negative weighted geometric-mean blend of the per-backbone OOF-validated heads.',
    'elapsed_hours': round((datetime.datetime.now() - start).total_seconds() / 3600, 2),
    'external_data': [f'timm/{m} (https://huggingface.co/timm/{m})' for m in models],
    'best_single_model_val': round(s['best_single'], 5),
    'n_models_in_final': len(used),
    'blend_weights': {k: round(v, 4) for k, v in s['weights'].items()},
}
json.dump(final, open(f'{ROOT}/work/final.json', 'w'), indent=1)
print(json.dumps(final, indent=1))
