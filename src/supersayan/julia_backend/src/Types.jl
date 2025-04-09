module Types

export LWE, convert_pyobject_to_lwe, convert_pyobjects_to_lwes

using PyCall

"""
    struct LWE

An LWE ciphertext consists of a mask (a vector of Float64) and a masked value (Float64).
"""
struct LWE
    mask::Vector{Float64}
    masked::Float64
end

"""
    convert_pyobject_to_lwe(py_obj::PyObject)

Converts a Python object with mask and masked attributes to an LWE struct.
"""
function convert_pyobject_to_lwe(py_obj::PyObject)
    return LWE(py_obj.mask, py_obj.masked)
end

"""
    convert_pyobjects_to_lwes(py_objs::Vector{PyObject})

Converts a vector of Python objects to a vector of LWE structs.
"""
function convert_pyobjects_to_lwes(py_objs::Vector{PyObject})
    return [convert_pyobject_to_lwe(obj) for obj in py_objs]
end

end
