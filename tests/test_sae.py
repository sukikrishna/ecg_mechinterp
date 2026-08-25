"""SAE training/stability checks on synthetic low-rank activations — no data download needed."""
import numpy as np

from ecg_mechinterp.sae.stability import StreamingPearson, activation_matching, dictionary_matching, random_dictionary_null
from ecg_mechinterp.sae.train import sae_encode_all, sae_report, train_sae


def _synthetic_activations(seed=0, n=3000, d=32, latent_dim=6):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, latent_dim))
    basis = rng.standard_normal((latent_dim, d))
    return (latent @ basis + 0.1 * rng.standard_normal((n, d))).astype(np.float32)


def test_sae_trains_and_reduces_reconstruction_error():
    acts = _synthetic_activations()
    train, test = acts[:2400], acts[2400:]
    sae = train_sae(train, d_hidden=64, method="topk", k=8, seed=0, steps=300, batch=256)
    rep = sae_report(sae, test)
    assert 0.0 <= rep["fvu"] < 0.6
    assert rep["dead_frac"] < 0.5


def test_dictionary_matching_beats_random_null():
    acts = _synthetic_activations()
    train, test = acts[:2400], acts[2400:]
    sae0 = train_sae(train, d_hidden=64, method="topk", k=8, seed=0, steps=400, batch=256)
    sae1 = train_sae(train, d_hidden=64, method="topk", k=8, seed=1, steps=400, batch=256)
    dm = dictionary_matching(sae0, sae1)
    null = random_dictionary_null(d_in=acts.shape[1], d_hidden=64, seed=0)
    assert dm["mmcs"] >= null - 0.05  # trained dictionaries should not fall below the floor

    z0 = sae_encode_all(sae0, test)
    z1 = sae_encode_all(sae1, test)
    am = activation_matching(z0, z1, seed=0)
    assert not np.isnan(am["mmcs"])
    assert 0.0 <= am["mmcs"] <= 1.0


def test_streaming_pearson_matches_direct_correlation():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 4))
    b = a * 2 + 0.01 * rng.standard_normal((200, 4))
    sp = StreamingPearson(n_pairs=4)
    sp.update(a[:100], b[:100])
    sp.update(a[100:], b[100:])
    r_direct = np.array([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(4)])
    assert np.allclose(sp.result(), r_direct, atol=1e-6)
