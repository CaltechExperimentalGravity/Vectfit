# Vectfit: Python Vector Fitting for Rational Function Approximation

`Vectfit` is a Python implementation of the Fast Relaxed Vector Fitting algorithm originally developed by Gustavsen & Semlyen, commonly used in electrical engineering and physics to approximate frequency-domain responses with rational functions. This repository, maintained by the Caltech Experimental Gravity group, provides a robust and accessible implementation suitable for graduate-level research and practical applications.

## What is Vector Fitting?

Vector Fitting (Vectfit) approximates a complex-valued frequency-domain function \( H(s) \) by a rational function:

\[
H(s) \approx \sum_{m=1}^{N} \frac{r_m}{s - p_m} + d + s e
\]

where:

- \( p_m \): Poles (complex frequencies).
- \( r_m \): Residues corresponding to each pole.
- \( d, e \): Constant and linear terms for proper/non-proper rational approximations.

Vectfit is efficient for modeling systems characterized by measured frequency response data, including electronic circuits, mechanical resonances, optical cavities, and gravitational-wave detectors.

## Features of this Implementation

- **Pure-Python** implementation (no compiled dependencies).
- Handles proper and non-proper rational approximations.
- Automatic initial pole placement, pole relocation, and iterative fitting.
- Supports weighted fitting to prioritize frequency ranges.
- Well-documented code and easy-to-follow examples.

## Repository Structure

```text
.
├── vectfit.py                # Main algorithm implementation
├── test_vectfit_auto.py      # Comprehensive example script
├── test_vectfit_NPRO.py      # Example for non-proper rational fits
├── test_vectfit_w_weight.py  # Weighted fitting example
├── data/                     # Example data files
├── Figures/                  # Output figures from example scripts
├── BodePlot.mplstyle         # Matplotlib style for clear Bode plots
├── LICENSE                   # GPL-2.0 License
└── README.md                 # This file
```

## Getting Started

### Requirements
- Python 3.x
- NumPy, SciPy, Matplotlib

Install with pip:

```bash
pip install numpy scipy matplotlib
```

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/CaltechExperimentalGravity/Vectfit.git
cd Vectfit
```

## Quick Example: Fitting Data with Vectfit

Suppose you have measured frequency-domain data (`freq`, `H_data`) and want to fit it:

```python
import numpy as np
import vectfit
import matplotlib.pyplot as plt

# Example frequency data (rad/s)
freq = np.linspace(1e2, 1e5, 500)

# Example measured response data
H_data = np.exp(-1j * freq * 1e-4) / (1 + 1j * freq * 1e-3)

# Initial pole guess (log-spaced complex poles)
initial_poles = vectfit.generate_initial_poles(freq, n_poles=10)

# Perform vector fitting
poles, residues, d, fit_result, rms_error = vectfit.vectfit(H_data, freq, initial_poles)

# Plot original vs fitted data
plt.figure()
plt.semilogx(freq, 20 * np.log10(np.abs(H_data)), label='Original Data')
plt.semilogx(freq, 20 * np.log10(np.abs(fit_result)), '--', label='Vectfit Approximation')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.title('Vector Fitting Example')
plt.grid(True, which='both')
plt.show()
```

## Choosing Initial Poles

Selecting initial poles significantly affects convergence:

- Use evenly spaced complex poles spanning your data range.
- Utilize `vectfit.generate_initial_poles(freq, n_poles)` for convenience.

## Weighted Fitting

Weights allow emphasis or suppression of specific frequency ranges:

```python
weights = 1 / np.abs(H_data)  # Example weight emphasizing weaker signals
poles, residues, d, fit_result, rms_error = vectfit.vectfit(H_data, freq, initial_poles, weights=weights)
```

## Non-Proper Rational Fitting

Non-proper rational fitting adds linear frequency dependence:

```python
poles, residues, d, e, fit_result, rms_error = vectfit.vectfit(H_data, freq, initial_poles, n_polynomial=1)
```

`n_polynomial=1` adds a linear frequency term (`e`) to the rational approximation.

## Testing and Validation

Use included scripts to test implementation and learn usage patterns:

- Basic test: `test_vectfit_auto.py`
- Weighted fitting example: `test_vectfit_w_weight.py`
- Non-proper rational example: `test_vectfit_NPRO.py`

Run any of these scripts directly from your terminal:

```bash
python test_vectfit_auto.py
```

## References

- **Primary Papers:**
  - B. Gustavsen and A. Semlyen, "Rational approximation of frequency domain responses by Vector Fitting," *IEEE Trans. Power Delivery*, vol. 14, no. 3, pp. 1052-1061, 1999.
  - B. Gustavsen, "Improving the pole relocating properties of vector fitting," *IEEE Trans. Power Delivery*, vol. 21, no. 3, pp. 1587-1592, 2006.

## License

Distributed under the [GPL-2.0 License](LICENSE).


