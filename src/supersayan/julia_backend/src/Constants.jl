module Constants

export n, σ_LWE, p, S

const n = 462                         # Number of components in the secret LWE-key
const σ_LWE = 2.0^-20                 # Standard deviation for LWE noise
const p = 5                           # Number of elements in the discrete torus T_p
const S = [-1, 1]                     # The subset of ℤ from which key components are picked

end
