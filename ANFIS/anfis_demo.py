"""
ANFIS training demonstration for PADS Section 6.3.1.

Implements a Type-3 (first-order Sugeno) ANFIS with 3 Gaussian membership
functions, matching the five-layer structure described in the report:
  Layer 1 - Gaussian fuzzification     mu_j(Y) = exp(-(Y-c_j)^2 / 2 sigma_j^2)
  Layer 2 - rule firing strength       w_j  (single input -> pass-through)
  Layer 3 - normalisation              wbar_j = w_j / sum(w)
  Layer 4 - first-order Sugeno         f_j = a_j*Y + b_j
  Layer 5 - weighted output            alpha = sum(wbar_j * f_j)

Trainable parameters: c_j, sigma_j (Layer 1) and a_j, b_j (Layer 4),
optimised by gradient descent to fit a designer-specified target curve.

NOTE: the target curve is SYNTHETIC, encoding domain knowledge only. This
demo verifies the training MECHANISM converges; operational fine-tuning
still requires a real luminance/trust-weighting dataset (see Section 6.5).
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Synthetic target curves (domain knowledge, NOT measured data)
#    Y in [0,1] normalised average luminance.
#    Rule 3 (dark)     -> low Y  -> trust Stream B (NIR+thermal)
#    Rule 2 (normal)   -> mid Y  -> trust Stream A (RGB+LiDAR)
#    Rule 1 (blinding) -> high Y -> Stream A degraded, trust B again
# ----------------------------------------------------------------------
def target_alpha_A(Y):
    # high in mid-luminance, falls off when dark OR blinding
    return 0.15 + 0.80 * np.exp(-((Y - 0.55) ** 2) / (2 * 0.18 ** 2))

def target_alpha_B(Y):
    # complementary: high when dark or blinding, low at normal daylight
    return 1.0 - target_alpha_A(Y) + 0.05

# ----------------------------------------------------------------------
# 2. ANFIS model (one instance per stream)
# ----------------------------------------------------------------------
class ANFIS:
    def __init__(self):
        # Layer 1 - 3 Gaussian MFs, initialised from domain knowledge
        # (dark / normal / blinding centres)
        self.c = np.array([0.15, 0.55, 0.90])
        self.sigma = np.array([0.18, 0.18, 0.18])
        # Layer 4 - first-order Sugeno consequents a_j*Y + b_j.
        # Initialised from domain knowledge: each rule's constant b_j is a
        # rough prior on that lighting regime's trust level; training then
        # refines both b_j and the (initially zero) slopes a_j.
        self.a = np.zeros(3)
        self.b = np.array([0.40, 0.75, 0.35])  # dark / normal / blinding prior

    def _forward(self, Y):
        # Y: array (N,). Returns alpha (N,) plus cached terms for backprop.
        Y = Y[:, None]                                   # (N,1)
        mu = np.exp(-((Y - self.c) ** 2) / (2 * self.sigma ** 2))  # (N,3)
        w = mu                                           # single input
        wsum = w.sum(axis=1, keepdims=True)
        wbar = w / wsum                                  # (N,3)
        f = self.a * Y + self.b                          # (N,3)
        alpha = (wbar * f).sum(axis=1)                   # (N,)
        cache = (Y, mu, w, wsum, wbar, f)
        return alpha, cache

    def predict(self, Y):
        return self._forward(np.asarray(Y, float))[0]

    def train(self, Y, t, epochs=400, lr=0.05):
        Y = np.asarray(Y, float)
        t = np.asarray(t, float)
        N = len(Y)
        history = []
        for ep in range(epochs):
            alpha, (Yc, mu, w, wsum, wbar, f) = self._forward(Y)
            err = alpha - t                              # (N,)
            mse = np.mean(err ** 2)
            history.append(mse)

            g = (2.0 / N) * err[:, None]                 # dL/dalpha (N,1)

            # consequents (Layer 4): dalpha/df_j = wbar_j
            grad_a = (g * wbar * Yc).sum(axis=0)
            grad_b = (g * wbar).sum(axis=0)

            # premise (Layer 1): dalpha/dwbar_j then dwbar/dmu then dmu/dc,sigma
            # dalpha/dwbar_j = f_j ; dwbar_j/dw_k = (delta_jk - wbar_k)/wsum
            dL_dwbar = g * f                             # (N,3)
            # propagate through normalisation
            dL_dw = (dL_dwbar / wsum) - (
                (dL_dwbar * wbar).sum(axis=1, keepdims=True) / wsum
            )                                            # (N,3)
            dmu_dc = mu * (Yc - self.c) / (self.sigma ** 2)
            dmu_ds = mu * (Yc - self.c) ** 2 / (self.sigma ** 3)
            grad_c = (dL_dw * dmu_dc).sum(axis=0)
            grad_sigma = (dL_dw * dmu_ds).sum(axis=0)

            self.a -= lr * grad_a
            self.b -= lr * grad_b
            self.c -= lr * grad_c
            self.sigma -= lr * grad_sigma
            self.sigma = np.clip(self.sigma, 0.03, None)  # keep MFs valid
        return history

# ----------------------------------------------------------------------
# 3. Generate synthetic training data and train both streams
# ----------------------------------------------------------------------
Y_train = rng.uniform(0, 1, 300)
tA = target_alpha_A(Y_train) + rng.normal(0, 0.02, Y_train.size)
tB = target_alpha_B(Y_train) + rng.normal(0, 0.02, Y_train.size)

anfisA, anfisB = ANFIS(), ANFIS()
# Stream B's domain prior is complementary to A: trust B when dark/blinding.
anfisB.b = np.array([0.65, 0.30, 0.70])  # dark / normal / blinding prior

# capture pre-training (domain-knowledge init) predictions
Y_plot = np.linspace(0, 1, 200)
preA = anfisA.predict(Y_plot)
preB = anfisB.predict(Y_plot)

histA = anfisA.train(Y_train, tA, epochs=400, lr=0.08)
histB = anfisB.train(Y_train, tB, epochs=400, lr=0.08)

postA = anfisA.predict(Y_plot)
postB = anfisB.predict(Y_plot)

print(f"Stream A  final train MSE: {histA[-1]:.2e}")
print(f"Stream B  final train MSE: {histB[-1]:.2e}")
print(f"Stream A  MSE reduction:   {histA[0]:.2e} -> {histA[-1]:.2e}")
print(f"Stream B  MSE reduction:   {histB[0]:.2e} -> {histB[-1]:.2e}")

# ----------------------------------------------------------------------
# 4. Plot - convergence panel + fitted-curve panel
# ----------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))

# Panel (a): fitted trust curves
ax[0].plot(Y_plot, target_alpha_A(Y_plot), color="navy", lw=2.0,
           label=r"Target $\alpha_A$")
ax[0].plot(Y_plot, target_alpha_B(Y_plot), color="darkred", lw=2.0,
           label=r"Target $\alpha_B$")
ax[0].plot(Y_plot, preA, color="navy", ls=":", lw=1.3,
           label=r"Init $\alpha_A$")
ax[0].plot(Y_plot, preB, color="darkred", ls=":", lw=1.3,
           label=r"Init $\alpha_B$")
ax[0].plot(Y_plot, postA, color="navy", ls="--", lw=1.6,
           label=r"Trained $\alpha_A$")
ax[0].plot(Y_plot, postB, color="darkred", ls="--", lw=1.6,
           label=r"Trained $\alpha_B$")
ax[0].set_xlabel(r"Average luminance $Y_{avg}$")
ax[0].set_ylabel(r"Stream trust weighting $\alpha$")
ax[0].set_title("(a) ANFIS fit to synthetic target")
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1.1)
ax[0].legend(fontsize=6.5, ncol=3, loc="upper center")
ax[0].grid(alpha=0.3)

# Panel (b): training convergence
ax[1].plot(histA, color="navy", lw=1.6, label="Stream A")
ax[1].plot(histB, color="darkred", lw=1.6, label="Stream B")
ax[1].set_yscale("log")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Training MSE")
ax[1].set_title("(b) Training convergence")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("anfis_training_demo.pdf", bbox_inches="tight")
plt.savefig("anfis_training_demo.png", dpi=160, bbox_inches="tight")
print("saved anfis_training_demo.pdf / .png")
