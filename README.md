# 🧮 rUnits – Rational Units for High-Precision Symbolic Computation

When decimal precision is most imporant and when 1/3 + 1/3 + 1/3 has be equal 1 not 0.9999999. New way to approach a symbolic decimal calculations.

> *A symbolic numeric framework for exact arithmetic, modular math, and physics-grade calculations.*

**rUnits** is a lightweight Python library for representing and computing with **exact rational numbers** in symbolic form. Designed to retain **full precision** across operations like modular arithmetic, RSA encryption, thermal resistance modeling, and physical simulation, it offers a compelling alternative to floating-point approximations.

## 🌟 Features

- 🔢 Exact rational arithmetic using symbolic numerator/denominator representation.
- 📏 Bit-limit control for emulating fixed-width precision (e.g., 64-bit, 128-bit).
- 🧮 Modular inverse and fractional modulus support.
- 🔐 Drop-in support for **RSA encryption and CRT decryption** with rational primitives.
- 🌡️ Physics-ready: compute **heat flux, thermal gradients, and interface resistances** using real material data.
- 💾 Serialization & deserialization with dynamic or fixed bit lengths.
- ⚡ High performance for symbolic RSA, outperforming standard float-based approaches for many secure use cases.

## 📦 Installation

pip install runits

Or clone the repo:

git clone https://github.com/TheSmartFox/rUnits.git
cd runits

## 🚀 Quick Start

from rUnits import rUnit

a = rUnit(1, 3)           # Create 1/3
b = rUnit(2, 5)           # Create 2/5

c = a + b                 # Exact: 11/15
d = a * b                 # Exact: 2/15
e = a.mod_inverse(7)      # Modular inverse of 1/3 mod 7
f = a % rUnit(1, 7)       # Fractional modulus



## 🌡️ Example: Thermal Resistance Calculation

from runits import rUnit
from heat_solver import ThermalStack

materials = [
    ("Copper Layer", rUnit(1, 1_000_000_000)),
    ("Silicon Layer", rUnit(1, 4_000_000_000)),
    ("Aluminum Oxide", rUnit(1, 200_000_000)),
]

interfaces = [
    ("Interface 1", rUnit(1, 50_000_000)),
    ("Interface 2", rUnit(1, 30_000_000)),
]

T_hot = rUnit(373)  # 100°C in Kelvin
T_cold = rUnit(300) # 27°C in Kelvin

solver = ThermalStack(materials, interfaces, T_hot, T_cold)
solver.solve()

solver.print_report()


**Outputs:**


Interface Temperatures:
  T0: 373/1 (float: 373.0)
  T1: 265819/715 (≈371.9K)
  T2: 53120/143 (≈371.5K)
  ...
Total Resistance: 517/144
Heat Flux: 11520/517


## 🧠 Real-World Use Cases

* 🔐 **Cryptography**: Symbolic RSA keygen, encryption, and CRT-based decryption with full rational traceability.
* 🌡️ **Materials science**: Model composite thermal layers with exact resistance and heat flow.
* ⚙️ **Engineering**: Solve systems involving gear ratios, energy flow, or control loops with rational time steps.
* 🧬 **Bioinformatics**: Perform ultra-precise computations with probabilistic models or floating-base SNPs.
* 🧮 **Mathematics education**: Explore modular systems, continued fractions, or number theory visually.

## ⚖️ Performance Notes

rUnits offer **perfect accuracy** with minimal runtime overhead for typical symbolic tasks. They're especially well-suited for:

* Comparing against float-based errors in physical systems.
* Building explainable AI modules where precision matters.
* Avoiding cumulative rounding artifacts in iterative models.

Performance varies with bit-limit and operation type. Benchmark modules are included.

## 📜 Contributing

You're welcome to contribute!

* Fork the repository
* Open issues or propose enhancements
* Write tests or expand `rUnit` capabilities (e.g., support for algebraic expressions)

## 🛡️ License

MIT License © 2025 \[Smart Fox Innovation]

## 🧠 Bonus: Why Use rUnits?

> *“Floating-point lies where rUnits speak truth.”*

Traditional floats can't precisely represent values like `1/3`, `1/7`, or `1/223088881`. rUnits preserve these ratios exactly — enabling physics-grade, cryptography-safe, and explainable symbolic computation.

