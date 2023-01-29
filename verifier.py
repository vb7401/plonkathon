import py_ecc.bn128 as b
from utils import *
from dataclasses import dataclass
from curve import *
from transcript import Transcript
from poly import Polynomial, Basis


@dataclass
class VerificationKey:
    """Verification key"""

    group_order: int
    # [q_M(x)]₁ (commitment to multiplication selector polynomial)
    Qm: G1Point
    # [q_L(x)]₁ (commitment to left selector polynomial)
    Ql: G1Point
    # [q_R(x)]₁ (commitment to right selector polynomial)
    Qr: G1Point
    # [q_O(x)]₁ (commitment to output selector polynomial)
    Qo: G1Point
    # [q_C(x)]₁ (commitment to constants selector polynomial)
    Qc: G1Point
    # [S_σ1(x)]₁ (commitment to the first permutation polynomial S_σ1(X))
    S1: G1Point
    # [S_σ2(x)]₁ (commitment to the second permutation polynomial S_σ2(X))
    S2: G1Point
    # [S_σ3(x)]₁ (commitment to the third permutation polynomial S_σ3(X))
    S3: G1Point
    # [x]₂ = xH, where H is a generator of G_2
    X_2: G2Point
    # nth root of unity, where n is the program's group order.
    w: Scalar
    # blinding
    blind: bool

    # More optimized version that tries hard to minimize pairings and
    # elliptic curve multiplications, but at the cost of being harder
    # to understand and mixing together a lot of the computations to
    # efficiently batch them.
    def verify_proof(self, group_order: int, pf, public=[]) -> bool:
        # skipping steps 1-3, sue me

        # step 4
        beta, gamma, alpha, zeta, v, u = self.compute_challenges(pf)

        # step 5
        zh_eval = zeta**group_order - Scalar(1)

        # step 6
        l1_eval = Polynomial(
            [Scalar(1)] + [Scalar(0)] * (group_order - 1),
            Basis.LAGRANGE
        ).barycentric_eval(zeta)

        # step 7
        PI = Polynomial(
            [Scalar(-val) for val in public]
            + [Scalar(0) for _ in range(self.group_order - len(public))],
            Basis.LAGRANGE,
        )
        PI_eval = PI.barycentric_eval(zeta)

        # step 8

        r_0 = (
            PI_eval - l1_eval * (alpha**2) - 
            (alpha * (pf.msg_4.a_eval + pf.msg_4.s1_eval * beta + gamma) *
            (pf.msg_4.b_eval + pf.msg_4.s2_eval * beta + gamma) *
            (pf.msg_4.c_eval + gamma) * pf.msg_4.z_shifted_eval)
        )

        # step 9
        d_1 = ec_lincomb(
            [
                (self.Qm, pf.msg_4.a_eval * pf.msg_4.b_eval),
                (self.Ql, pf.msg_4.a_eval),
                (self.Qr, pf.msg_4.b_eval),
                (self.Qo, pf.msg_4.c_eval),
                (self.Qc, Scalar(1)),
                (
                    pf.msg_2.z_1,
                    (pf.msg_4.a_eval + zeta * beta + gamma) *
                    (pf.msg_4.b_eval + zeta * 2 * beta + gamma) *
                    (pf.msg_4.c_eval + zeta * 3 * beta + gamma) * alpha +
                    l1_eval * (alpha**2) + u
                ),
                (
                    self.S3, 
                    -(pf.msg_4.a_eval + pf.msg_4.s1_eval * beta + gamma) *
                     (pf.msg_4.b_eval + pf.msg_4.s2_eval * beta + gamma) *
                     alpha * beta * pf.msg_4.z_shifted_eval
                ),
                (pf.msg_3.t_lo_1, -zh_eval),
                (pf.msg_3.t_mid_1, -(zh_eval)*(zeta**(group_order))),
                (pf.msg_3.t_hi_1, -(zh_eval)*(zeta**(2*group_order)))
            ]
        )

        # step 10
        f_1 = ec_lincomb(
            [
                (d_1, 1),
                (pf.msg_1.a_1, v),
                (pf.msg_1.b_1, v**2),
                (pf.msg_1.c_1, v**3),
                (self.S1, v**4),
                (self.S2, v**5)
            ]
        )

        # step 11
        e_1 = ec_mul(
            b.G1,
            (-r_0 + v * pf.msg_4.a_eval + v**2 * pf.msg_4.b_eval + v**3* pf.msg_4.c_eval
                + v**4 * pf.msg_4.s1_eval + v**5 * pf.msg_4.s2_eval + u * pf.msg_4.z_shifted_eval)
        )

        assert b.pairing(
            self.X_2,
            ec_lincomb(
                [
                    (pf.msg_5.W_z_1, Scalar(1)),
                    (pf.msg_5.W_zw_1, u),
                ]
            )
        ) == b.pairing(
            b.G2,
            ec_lincomb(
                [
                    (pf.msg_5.W_z_1, zeta),
                    (pf.msg_5.W_zw_1, u * zeta * Scalar.root_of_unity(group_order)),
                    (f_1, Scalar(1)),
                    (e_1, Scalar(-1))
                ]
            )
        )

        return True

    # Basic, easier-to-understand version of what's going on.
    # Feel free to use multiple pairings.
    def verify_proof_unoptimized(self, group_order: int, pf, public=[]) -> bool:

        return False

    # Compute challenges (should be same as those computed by prover)
    def compute_challenges(
        self, proof
    ) -> tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
        transcript = Transcript(b"plonk")
        beta, gamma = transcript.round_1(proof.msg_1)
        alpha, _fft_cofactor = transcript.round_2(proof.msg_2)
        zeta = transcript.round_3(proof.msg_3)
        v = transcript.round_4(proof.msg_4)
        u = transcript.round_5(proof.msg_5)

        return beta, gamma, alpha, zeta, v, u
