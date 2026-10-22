"""Synthetic source-contract audit; no pretrained weights or market accuracy claim.

Run with the TW project's environment and both source trees on PYTHONPATH.
"""
import json
from datetime import date

import numpy as np
import torch
from timesfm3.torch.configs import ResidualBlockConfig, StackedTransformersConfig, TransformerConfig
from timesfm3.torch.model import TimesFM3Torch
from tw_scenario.quant.timesfm_head_training import HeadSample, _batch_inputs, _one_step_logits


torch.manual_seed(4655)
rng = np.random.default_rng(4655)
model = TimesFM3Torch(
    input_patch_len=32, output_patch_len=64,
    residual_block_config=ResidualBlockConfig(hidden_dims=32, output_dims=32, use_bias=False, activation="relu"),
    transformer_config=StackedTransformersConfig(num_layers=2,
        transformer=TransformerConfig(model_dims=32, hidden_dims=32, num_heads=4,
                                      use_rope_var=False, deterministic=True, attention_norm="rms",
                                      feedforward_norm="rms", qk_norm="rms", use_bias=False,
                                      use_rope_seq=True, ff_activation="relu")),
    use_linear_detrending=False,
)
model.eval()
history = np.stack([np.linspace(i, i + 4, 64) for i in range(4)], axis=1).astype('float32')
past = rng.normal(size=(64, 27)).astype('float32')
future = rng.normal(size=27).astype('float32')
sample = HeadSample(day=date(2026, 1, 2), history=history, past_features=past,
                    forecast_features=future, target=np.zeros(4, dtype='float32'),
                    previous_close=100., actual={}, contract='')
inputs, cpm, patches, targets = _batch_inputs(model, [sample], np.zeros(27), np.ones(27))
with torch.no_grad():
    custom = _one_step_logits(model, inputs, cpm, patches, targets)
    official = model.decode(torch.tensor(history.T[None]), horizon=1,
                            past_future_covariates=torch.tensor(
                                np.concatenate([past, future[None]], axis=0).T[None]))[:, :4, 0]
    parity = float((custom - official).abs().max())
    assert parity < 1e-5, parity
    model.use_linear_detrending = True
    detrended = model.decode(torch.tensor(history.T[None]), horizon=1,
                            past_future_covariates=torch.tensor(
                                np.concatenate([past, future[None]], axis=0).T[None]))[:, :4, 0]
    detrend_difference = float((custom - detrended).abs().max())
    assert detrend_difference > 1e-5
    changed = {k: v.clone() for k, v in inputs.items()}
    changed['values'][:, :4, patches:] = 999.
    masked_difference = float((_one_step_logits(model, changed, cpm, patches, targets) - custom).abs().max())
    assert masked_difference == 0.
    changed = {k: v.clone() for k, v in inputs.items()}
    changed['values'][:, 4:, patches, 0] += 3.
    covariate_difference = float((_one_step_logits(model, changed, cpm, patches, targets) - custom).abs().max())
    assert covariate_difference > 0.
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith('output_head.'))
_one_step_logits(model, inputs, cpm, patches, targets).square().mean().backward()
with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
assert set(with_grad) == {'output_head.weight', 'output_head.bias'}
assert all(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
           for n, p in model.named_parameters() if n in with_grad)
print(json.dumps({'scope':'synthetic small model, not pretrained accuracy',
                  'status':'PASS', 'custom_vs_decode_detrend_disabled_max_abs':parity,
                  'custom_vs_decode_detrend_enabled_max_abs':detrend_difference,
                  'masked_future_target_change_max_abs':masked_difference,
                  'known_future_covariate_change_max_abs':covariate_difference,
                  'gradient_parameter_names':with_grad},indent=2))
