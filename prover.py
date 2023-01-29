from compiler.program import Program, CommonPreprocessedInput
from utils import *
from setup import *
from typing import Optional
from dataclasses import dataclass
from transcript import Transcript, Message1, Message2, Message3, Message4, Message5
from poly import Polynomial, Basis


@dataclass
class Proof:
    msg_1: Message1
    msg_2: Message2
    msg_3: Message3
    msg_4: Message4
    msg_5: Message5

    def flatten(self):
        proof = {}
        proof["a_1"] = self.msg_1.a_1
        proof["b_1"] = self.msg_1.b_1
        proof["c_1"] = self.msg_1.c_1
        proof["z_1"] = self.msg_2.z_1
        proof["t_lo_1"] = self.msg_3.t_lo_1
        proof["t_mid_1"] = self.msg_3.t_mid_1
        proof["t_hi_1"] = self.msg_3.t_hi_1
        proof["a_eval"] = self.msg_4.a_eval
        proof["b_eval"] = self.msg_4.b_eval
        proof["c_eval"] = self.msg_4.c_eval
        proof["s1_eval"] = self.msg_4.s1_eval
        proof["s2_eval"] = self.msg_4.s2_eval
        proof["z_shifted_eval"] = self.msg_4.z_shifted_eval
        proof["W_z_1"] = self.msg_5.W_z_1
        proof["W_zw_1"] = self.msg_5.W_zw_1
        return proof


@dataclass
class Prover:
    group_order: int
    setup: Setup
    program: Program
    pk: CommonPreprocessedInput

    def __init__(self, setup: Setup, program: Program):
        self.group_order = program.group_order
        self.setup = setup
        self.program = program
        self.pk = program.common_preprocessed_input()

    def prove(self, witness: dict[Optional[str], int]) -> Proof:
        # Initialise Fiat-Shamir transcript
        transcript = Transcript(b"plonk")

        # Collect fixed and public information
        # FIXME: Hash pk and PI into transcript
        public_vars = self.program.get_public_assignments()
        PI = Polynomial(
            [Scalar(-witness[v]) for v in public_vars]
            + [Scalar(0) for _ in range(self.group_order - len(public_vars))],
            Basis.LAGRANGE,
        )
        self.PI = PI

        # Round 1
        msg_1 = self.round_1(witness)
        self.beta, self.gamma = transcript.round_1(msg_1)

        # Round 2
        msg_2 = self.round_2()
        self.alpha, self.fft_cofactor = transcript.round_2(msg_2)

        # Round 3
        msg_3 = self.round_3()
        self.zeta = transcript.round_3(msg_3)

        # Round 4
        msg_4 = self.round_4()
        self.v = transcript.round_4(msg_4)

        # Round 5
        msg_5 = self.round_5()

        return Proof(msg_1, msg_2, msg_3, msg_4, msg_5)

    def round_1(
        self,
        witness: dict[Optional[str], int],
    ) -> Message1:
        program = self.program
        setup = self.setup
        group_order = self.group_order

        if None not in witness:
            witness[None] = 0

        A_i, B_i, C_i = [Scalar(0)] * group_order, [Scalar(0)] * group_order, [Scalar(0)] * group_order
        for i, wires in enumerate(program.wires()):
            A_i[i] = Scalar(witness[wires.L])
            B_i[i] = Scalar(witness[wires.R])
            C_i[i] = Scalar(witness[wires.O])

        self.A = Polynomial(A_i, Basis.LAGRANGE)
        self.B = Polynomial(B_i, Basis.LAGRANGE)
        self.C = Polynomial(C_i, Basis.LAGRANGE)

        # Sanity check that witness fulfils gate constraints
        assert (
            self.A * self.pk.QL
            + self.B * self.pk.QR
            + self.A * self.B * self.pk.QM
            + self.C * self.pk.QO
            + self.PI
            + self.pk.QC
            == Polynomial([Scalar(0)] * group_order, Basis.LAGRANGE)
        ), "witnes does not satisfy gate constraints"

        a_1, b_1, c_1 = setup.commit(self.A), setup.commit(self.B), setup.commit(self.C)

        # Return a_1, b_1, c_1
        return Message1(a_1, b_1, c_1)

    def round_2(self) -> Message2:
        group_order = self.group_order
        setup = self.setup        
        roots_of_unity = Scalar.roots_of_unity(group_order)

        Z_values = [Scalar(1)] * (group_order+1)
        for i in range(1, group_order+1):
            Z_values[i] = Z_values[i-1] * (
                self.rlc(self.A.values[i-1], roots_of_unity[i-1]) *
                self.rlc(self.B.values[i-1], 2*roots_of_unity[i-1]) *
                self.rlc(self.C.values[i-1], 3*roots_of_unity[i-1])
            ) / (
                self.rlc(self.A.values[i-1], self.pk.S1.values[i-1]) *
                self.rlc(self.B.values[i-1], self.pk.S2.values[i-1]) *
                self.rlc(self.C.values[i-1], self.pk.S3.values[i-1])
            )

        # Check that the last term Z_n = 1
        assert Z_values.pop() == 1

        # Sanity-check that Z was computed correctly
        for i in range(group_order):
            assert (
                self.rlc(self.A.values[i], roots_of_unity[i])
                * self.rlc(self.B.values[i], 2 * roots_of_unity[i])
                * self.rlc(self.C.values[i], 3 * roots_of_unity[i])
            ) * Z_values[i] - (
                self.rlc(self.A.values[i], self.pk.S1.values[i])
                * self.rlc(self.B.values[i], self.pk.S2.values[i])
                * self.rlc(self.C.values[i], self.pk.S3.values[i])
            ) * Z_values[
                (i + 1) % group_order
            ] == 0

        self.Z = Polynomial(Z_values, Basis.LAGRANGE)
        z_1 = setup.commit(self.Z)

        # Return z_1
        return Message2(z_1)

    def round_3(self) -> Message3:
        group_order = self.group_order
        setup = self.setup
        fft_cofactor = self.fft_cofactor

        # Expand all polynomials into the coset extended Lagrange basis
        L0_big = self.fft_expand(
            Polynomial([Scalar(1)] + [Scalar(0)] * (group_order - 1), Basis.LAGRANGE)
        )
        ZH_big = Polynomial(
            [Scalar(-1)] +
            [Scalar(0)] * (group_order - 1) +
            [fft_cofactor**group_order] +
            [Scalar(0)] * (group_order*3 - 1),
            Basis.MONOMIAL
        ).fft()
        X_big = self.fft_expand(
            Polynomial(Scalar.roots_of_unity(group_order), Basis.LAGRANGE)
        )
        a_big = self.fft_expand(self.A)
        b_big = self.fft_expand(self.B)
        c_big = self.fft_expand(self.C)
        Qm_big = self.fft_expand(self.pk.QM)
        Ql_big = self.fft_expand(self.pk.QL)
        Qr_big = self.fft_expand(self.pk.QR)
        Qo_big = self.fft_expand(self.pk.QO)
        Qc_big = self.fft_expand(self.pk.QC)
        PI_big = self.fft_expand(self.PI)
        S1_big = self.fft_expand(self.pk.S1)
        S2_big = self.fft_expand(self.pk.S2)
        S3_big = self.fft_expand(self.pk.S3)
        Z_big = self.fft_expand(self.Z)
        Zw_big = self.fft_expand(Polynomial(
            self.Z.values[1:] + [self.Z.values[0]],
            Basis.LAGRANGE
        ))

        # constraint check
        T_constraint = (
            a_big * b_big * Qm_big +
            a_big * Ql_big +
            b_big * Qr_big +
            c_big * Qo_big +
            PI_big +
            Qc_big
        );

        # grand product computation check
        T_gp_comp = (
            (
                self.rlc_poly(a_big, X_big) *
                self.rlc_poly(b_big, X_big*Scalar(2)) *
                self.rlc_poly(c_big, X_big*Scalar(3)) *
                Z_big
            ) - (
                self.rlc_poly(a_big, S1_big) *
                self.rlc_poly(b_big, S2_big) *
                self.rlc_poly(c_big, S3_big) *
                Zw_big
            )
        ) * self.alpha;

        # grand product is equal check
        T_gp_equal = (
            (Z_big - Scalar(1)) * L0_big
        ) * self.alpha * self.alpha;

        # division
        QUOT_big = (T_constraint + T_gp_comp + T_gp_equal) / ZH_big;
        T_coeffs = self.expanded_evals_to_coeffs(QUOT_big).values
        
        # Sanity check: QUOT has degree < 3n
        assert (
            T_coeffs[-group_order:]
            == [0] * group_order
        )
        print("Generated the quotient polynomial")

        # Compute T1, T2, T3
        self.T1 = Polynomial(T_coeffs[:group_order], Basis.MONOMIAL)
        self.T2 = Polynomial(T_coeffs[group_order:2*group_order], Basis.MONOMIAL)
        self.T3 = Polynomial(T_coeffs[2*group_order:3*group_order], Basis.MONOMIAL)

        # Sanity check that we've computed T1, T2, T3 correctly
        assert (
            self.T1.standard_eval(fft_cofactor)
            + self.T2.standard_eval(fft_cofactor) * fft_cofactor**group_order
            + self.T3.standard_eval(fft_cofactor) * fft_cofactor**(group_order * 2)
        ) == QUOT_big.values[0]

        print("Generated T1, T2, T3 polynomials")

        t_lo_1 = setup.commit_monomial(self.T1)
        t_mid_1 = setup.commit_monomial(self.T2)
        t_hi_1 = setup.commit_monomial(self.T3)

        # Return t_lo_1, t_mid_1, t_hi_1
        return Message3(t_lo_1, t_mid_1, t_hi_1)

    def round_4(self) -> Message4:
        zeta = self.zeta 

        self.a_eval = self.A.barycentric_eval(zeta)
        self.b_eval = self.B.barycentric_eval(zeta)
        self.c_eval = self.C.barycentric_eval(zeta)
        self.s1_eval = self.pk.S1.barycentric_eval(zeta)
        self.s2_eval = self.pk.S2.barycentric_eval(zeta)
        self.z_shifted_eval = self.Z.barycentric_eval(zeta * Scalar.root_of_unity(self.group_order))

        # Return a_eval, b_eval, c_eval, s1_eval, s2_eval, z_shifted_eval
        return Message4(self.a_eval, self.b_eval, self.c_eval, self.s1_eval, self.s2_eval, self.z_shifted_eval)

    def round_5(self) -> Message5:
        zeta = self.zeta
        v = self.v

        # more zeta evaluations
        zh_eval = Polynomial( 
            [Scalar(-1)] +
            [Scalar(0)] * (self.group_order - 1) +
            [Scalar(1)],
            Basis.MONOMIAL
        ).standard_eval(zeta)

        l1_eval = Polynomial(
            [Scalar(1)] + [Scalar(0)] * (self.group_order - 1),
            Basis.LAGRANGE
        ).barycentric_eval(zeta)

        # compute constraint part of R
        R_constraint = (
            self.pk.QM * self.a_eval * self.b_eval +
            self.pk.QL * self.a_eval +
            self.pk.QR * self.b_eval +
            self.pk.QO * self.c_eval +
            self.PI + 
            self.pk.QC
        )

        # compute grand product check
        R_gp_comp = (
            ((self.Z) * 
                self.rlc(self.a_eval, zeta) *
                self.rlc(self.b_eval, 2 * zeta) *  
                self.rlc(self.c_eval, 3 * zeta)) - 
            ((self.pk.S3 * self.beta + self.c_eval + self.gamma) *
                self.z_shifted_eval *
                self.rlc(self.a_eval, self.s1_eval) *
                self.rlc(self.b_eval, self.s2_eval))
        ) * self.alpha

        # compute grand product is equal check
        R_gp_equal = (self.Z - Scalar(1)) * l1_eval * self.alpha * self.alpha

        # compute vanishing section
        R_vanishing = (self.T1 + 
            self.T2 * (zeta**self.group_order) +
            self.T3 * (zeta**(2 * self.group_order))) * zh_eval

        R = R_constraint + R_gp_comp + R_gp_equal - R_vanishing.fft()

        # Sanity-check R
        assert R.barycentric_eval(zeta) == 0

        print("Generated linearization polynomial R")

        # compute sum of all checks
        check_sum = (R + 
            (self.A - self.a_eval) * v +
            (self.B - self.b_eval) * v**2 +
            (self.C - self.c_eval) * v**3 + 
            (self.pk.S1 - self.s1_eval) * v**4 +
            (self.pk.S2 - self.s2_eval) * v**5
        )
        m1 = Polynomial(
            [-self.zeta] + [Scalar(1)] + [Scalar(0)] * (self.group_order-2),
            Basis.MONOMIAL
        ).fft()
        m2 = Polynomial(
            [-self.zeta*Scalar.root_of_unity(self.group_order)] + [Scalar(1)] + [Scalar(0)] * (self.group_order-2),
            Basis.MONOMIAL
        ).fft()
        W_z = self.fft_expand(check_sum) / self.fft_expand(m1)
        W_zw = self.fft_expand(self.Z - self.z_shifted_eval) / self.fft_expand(m2)

        print("Generated final quotient witness polynomials")

        W_z_1 = self.setup.commit(self.expanded_evals_to_coeffs(W_z))
        W_zw_1 = self.setup.commit(self.expanded_evals_to_coeffs(W_zw))
        
        # Return W_z_1, W_zw_1
        return Message5(W_z_1, W_zw_1)

    def fft_expand(self, x: Polynomial):
        return x.to_coset_extended_lagrange(self.fft_cofactor)

    def expanded_evals_to_coeffs(self, x: Polynomial):
        return x.coset_extended_lagrange_to_coeffs(self.fft_cofactor)

    def rlc(self, term_1, term_2):
        return term_1 + term_2 * self.beta + self.gamma

    def rlc_poly(self, poly_1: Polynomial, poly_2: Polynomial):
        return poly_1 + poly_2 * self.beta + self.gamma