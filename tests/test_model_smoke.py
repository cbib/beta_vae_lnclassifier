"""Forward and loss smoke test for the two models, on CPU with random inputs.

Protects: output shapes, 58 attention positions for 15,000 nt, one token per
registry subgroup, the loss terms, and the absence of the removed adversarial,
contrastive and token-attention parts (including loading old checkpoints).
"""
import pytest
import torch

from data.feature_registry import REGISTRY
from models.beta_vae_subgroup import BetaVAESubgroup
from models.feature_only_classifier import FeatureOnlyClassifier
from trainers.beta_vae_subgroup_trainer import BetaVAESubgroupLoss

B, L, HEADS = 2, 15000, 4
REMOVED = ("length_head", "lambda_adv", "token_sa", "gradient_reversal", "contrast")


def _features():
    return dict(te_genomic=torch.randn(B, REGISTRY.te_dim),
                te_processed=torch.randn(B, REGISTRY.te_dim),
                nonb_genomic=torch.randn(B, REGISTRY.nonb_dim),
                nonb_processed=torch.randn(B, REGISTRY.nonb_dim),
                nonb2=torch.randn(B, REGISTRY.nonb2_dim))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BetaVAESubgroup(num_classes=2, input_dim=5, input_length=L, latent_dim=16,
                           cse_d_model=16, d_proj=16, attn_heads=HEADS,
                           classifier_hidden=[16], block_sources={"nonb": "genomic"})


@pytest.fixture(scope="module")
def output(model):
    model.eval()
    with torch.no_grad():
        return model(torch.rand(B, 5, L), deterministic=True, **_features())


def test_output_shapes(model, output):
    assert output["logits"].shape == (B, 2)
    assert output["mu"].shape == output["logvar"].shape == (B, 16)
    assert output["reconstruction"].shape[:2] == (B, 5)
    for block in REGISTRY.block_names:
        assert output[f"feat_logits_{block}"].shape == (B, 2)


def test_58_positions_and_one_token_per_subgroup(model, output):
    assert model.encoded_length == 58
    assert model.n_tokens == REGISTRY.total_tokens == 20
    assert output["attn_weights"].shape == (B, HEADS, 58, 20)


def test_loss_terms_present_and_finite(output):
    labels = torch.tensor([0, 1])
    feat = {b: output[f"feat_logits_{b}"] for b in REGISTRY.block_names}
    total, terms = BetaVAESubgroupLoss()(
        torch.rand_like(output["reconstruction"]), output["reconstruction"],
        output["mu"], output["logvar"], output["logits"], labels,
        attn_weights=output["attn_weights"], feat_logits=feat)
    expected = {"loss", "reconstruction", "kl", "classification",
                *(f"feat_{b}" for b in REGISTRY.block_names)}
    assert set(terms) == expected
    assert all(torch.isfinite(torch.tensor(float(v))) for v in terms.values())
    assert torch.isfinite(total)


def test_backward_runs(model):
    model.train()
    out = model(torch.rand(B, 5, L), **_features())
    (out["logits"].sum() + out["mu"].sum()).backward()
    model.zero_grad()


def test_no_removed_parts_in_the_model(model):
    names = list(model.state_dict()) + [n for n, _ in model.named_modules()]
    assert not [n for n in names if any(r in n for r in REMOVED)]


def test_old_checkpoint_keys_are_dropped_on_load(model):
    state = dict(model.state_dict())
    state["length_head.0.weight"] = torch.zeros(3, 3)
    state["lambda_adv"] = torch.tensor(0.1)
    model.load_state_dict(state)                      # strict, must not raise


def test_genuinely_missing_key_still_raises(model):
    state = dict(model.state_dict())
    state.pop(next(iter(state)))
    with pytest.raises(RuntimeError):
        model.load_state_dict(state)


def test_feature_only_model_forward():
    torch.manual_seed(0)
    net = FeatureOnlyClassifier(num_classes=2, d_proj=16, attn_heads=HEADS,
                                classifier_hidden=[16], block_sources={"nonb": "genomic"})
    net.eval()
    with torch.no_grad():
        out = net(deterministic=True, **_features())
    assert out["logits"].shape == (B, 2)
    assert not [n for n in net.state_dict() if any(r in n for r in REMOVED)]
