module Constants

import SupersayanTFHE.Types: SIGMA, P

const n = Int32(462)                  # Number of components in the secret LWE-key
const sigma = SIGMA(2.0^-20)          # Standard deviation for LWE noise (32-bit)
const p = P(128)                       # Number of elements in the discrete torus T_p (64-bit)
const S = Int32[-1, 1]                # The subset of ℤ from which key components are picked

export n, sigma, p, S

end
