from rUnits import rUnit
import math 
from math import sqrt

def test_rUnit_sum_series(n, bit_limit=256):
    """
    Test that sums terms in the form 1/i for i in range 1 to n using rUnit.

    Args:
        n (int): The maximum value of i in the series.
        bit_limit (int): The bit limit for the rUnit class.

    Returns:
        rUnit: The sum of the series as an rUnit.
    """
    total = rUnit(0, 1, bit_limit=bit_limit)  # Initialize with 0
    for i in range(1, n + 1, 1):
        term = rUnit(1, i, bit_limit=bit_limit)  # Create 1/i as an rUnit
        total += term  # Add the term to the total

    print(f"Sum of the series 1/1 + 1/2 + ... + 1/{n}: {total}")
    print(f"Floating-point equivalent: {total.to_float()}")

    return total

def calculate_gear_ratio(stages, bit_limit=64):
    """
    Calculate the overall gear ratio for a multi-stage gearbox using rUnit.

    Args:
        stages (list of tuples): Each tuple represents a gear stage as (N, D),
                                 where N is the driving gear teeth, and D is the driven gear teeth.
        bit_limit (int): The bit limit for the rUnit class.

    Returns:
        rUnit: The overall gear ratio as an rUnit.
    """
    overall_ratio = rUnit(1, 1, bit_limit=bit_limit)  # Start with a ratio of 1:1

    for N, D in stages:
        stage_ratio = rUnit(N, D, bit_limit=bit_limit)
        overall_ratio *= stage_ratio

    print(f"Overall Gear Ratio: {overall_ratio} (float: {overall_ratio.to_float()})")
    return overall_ratio



def sqrt_rUnit(value: rUnit):
    if value.numerator < 0:
        raise ValueError("Cannot compute square root of a negative rUnit.")
    sqrt_value = math.sqrt(value.to_float())  # Fallback to float for sqrt
    return rUnit.from_float(sqrt_value, value.bit_limit)


def compute_orbital_velocity(altitude_km):
    G = rUnit(667430000000, 10**13, bit_limit=512)  # Gravitational constant
    M = rUnit(5972000000000000000000000, 1, bit_limit=512)  # Earth's mass
    Re = rUnit(6371000, 1, bit_limit=512)  # Earth's radius in meters
    h = rUnit(altitude_km * 1000, 1, bit_limit=512)  # Altitude in meters

    print(f"G: {G}")
    print(f"M: {M}")
    print(f"Re: {Re}")
    print(f"h: {h}")
    
    
    # Orbital radius
    r = Re + h

    print(f"r: {r}")
    
    # GM term
    GM = G * M
    
    print(f"GM: {GM}")

    # Velocity squared (v^2)
    v_squared = GM / r
    
    print(f"GM/r: {v_squared}")


    # Approximate square root using built-in math.sqrt
    v = v_squared.sqrt()
    
    print(f"v: {v_squared}")
    
    return v


def selfTest():
    a = rUnit(1, 3)  # Create rUnit 1/3
    b = 2  # Integer

    # Perform addition
    c = a + b
    print(f"a + b: {c}")  # Expected: 7/3

    d = 5.5 + a
    print(f"d: {d}")  # Expected: ~5.833333...

    a = rUnit(3, 1, bit_limit=64)
    modulus = rUnit(7, 1, bit_limit=64)
    result = a.mod_inverse(modulus)
    print(f"Modular Inverse of {a} mod {modulus} = {result} (float: {result.to_float()})")


    r = rUnit(123, 456)
    serialized = r.serialize()
    # Output: [0, 123, 456]

    deserialized = (rUnit(0,1)).deserialize(serialized)
    # rUnit: 123/456 (bit_limit=64)
    
    print(f"serialization - deserialized: {r == deserialized}")

    r = rUnit(12345678901234567890, 98765432109876543210, 0)
    serialized = r.serialize()
    # Output: [1, 0x384048, 12345678901234567890, 98765432109876543210]

    deserialized = (rUnit(0,1)).deserialize(serialized)
    # rUnit: 12345678901234567890/98765432109876543210 (bit_limit=128)

    print(f"serialization - deserialized dynamic: {r == deserialized}")
    
    # Testing floor division
    a = rUnit(1, 3)  # 7/3
    b = rUnit(1, 223088881)  # 2
    result = a % b
    print(f"{a} % {b} = {result} (float: {result.to_float()})")


    # Testing floor division
    a = rUnit(7, 3)  # 7/3
    b = rUnit(2, 1)  # 2
    result = a // b
    print(f"{a} // {b} = {result} (float: {result.to_float()})")

    # Initialize rUnit
    base = rUnit(2, 3, bit_limit=64)
    exponent = 3

    # Compute power
    result = base.pow(exponent)
    print(f"{base} ** {exponent} = {result}")

    # Initialize rUnits
    base = rUnit(2, 3, bit_limit=64)
    exponent = 5
    modulus = rUnit(7, 1, bit_limit=64)

    # Compute modular power
    result = base.mod_pow(exponent, modulus)
    print(f"({base} ** {exponent}) % {modulus} = {result}")

    # Initialize with a 16-bit limit
    a = rUnit(2, 3, bit_limit=64)
    b = rUnit(1, 3, bit_limit=64)

    # Perform arithmetic
    c = a + b
    print(f"{a} + {b} = {c} (float: {c.to_float()})")


    a = rUnit(2**15, 1, bit_limit=16)  # Max numerator for 16 bits
    b = rUnit(1, 1, bit_limit=16)
    c = a + b
    print(f"{a} + {b} = {c} (float: {c.to_float()})")


    a = rUnit(123456, 789012, bit_limit=32)
    b = rUnit(987654, 321098, bit_limit=32)
    c = a * b
    print(f"{a} * {b} = {c} (float: {c.to_float()})")


    # Test zero denominator handling
    a = rUnit(1, 0)  # Should become 0/1
    print(f"a: {a} (float: {a.to_float()})")  # Expected: 0/1, 0.0

    # Test division by zero
    b = rUnit(1, 1)
    c = rUnit(0, 1)  # Represents zero
    result = b / c
    print(f"b / c: {result} (float: {result.to_float()})")  # Expected: 0/1, 0.0

    # Test addition with zero denominator
    d = rUnit(1, 0)
    e = rUnit(2, 3)
    f = d + e
    print(f"d + e: {f} (float: {f.to_float()})")  # Expected: 2/3, 0.666...

   # Test addition with zero denominator
   
    a = rUnit(1, 312, bit_limit=0)
    b = rUnit(1, 7652213000000000000000000000691, bit_limit=0)
    c = a * b
    print(f"a * b: {c.numerator}/{c.denominator} is_dynamic base (float: {c.to_float()}) c: {c}")
  
    a = rUnit(1, 1, bit_limit=0)
    b = rUnit(1, 12, bit_limit=0)
    c = a * b
    print(f"a * b: {c.numerator}/{c.denominator} is_dynamic base (float: {c.to_float()}) c: {c}")


    a0 = rUnit(1, 1,256)
    a = rUnit(1, 2,256)
    b = rUnit(1, 3,256)
    c = rUnit(1, 5,256)
    d = rUnit(1, 7,256)
    e = rUnit(1, 11,256)
    f = rUnit(1, 13,256)
    g = rUnit(1, 17,256)    
    h = rUnit(1, 19,256)    
    i = rUnit(1, 23,256)
    res = a0+a+b+c+d+e+f+g+h+i
    print(f"a0 + a + b + c + d + e + f + g + h + i: {res} (float: {res.to_float()})")  # Expected: ?


    a = rUnit(1, 3)
    b = rUnit(1, 6)
    c = a * b
    print(f"a * b: {c.numerator}/{c.denominator} (float: {c.to_float()})")


    a = rUnit(1, 3)
    b = 123456
    c = a + b
    print(f"a + b (int): {c.numerator}/{c.denominator} (float: {c.to_float()})")
    
    
    a = rUnit(1, 3)
    b = 3
    c = a * b
    print(f"a * b (int): {c.numerator}/{c.denominator} (float: {c.to_float()})")
    
    
    a = rUnit(0, 1)
    b = rUnit(1, 1)
    c = a * b
    print(f"a * b: {c.numerator}/{c.denominator} (float: {c.to_float()})")

    a = rUnit(1, 1)
    b = rUnit(1, 1)
    c = a * b
    print(f"a * b: {c.numerator}/{c.denominator} (float: {c.to_float()})")

    a = rUnit(0, 1)
    b = rUnit(1, 1)
    c = a + b
    print(f"a + b: {c.numerator}/{c.denominator} (float: {c.to_float()})")

    a = rUnit(0, 1)
    b = rUnit(1, 1)
    c = a / b
    print(f"a / b: {c.numerator}/{c.denominator} (float: {c.to_float()})")


    a = rUnit(0, 1)
    b = rUnit(1, 1)
    c = a / b
    print(f"a / b: {c.numerator}/{c.denominator} (float: {c.to_float()})")

    a = rUnit(3, 4)
    b = rUnit(2, 3)

    # rUnit * rUnit
    print(a * b)  # Output: 6/12 (bit_limit=64) (float: 0.5)

    # float * rUnit
    print(2.0 * a)  # Output: 3/2 (bit_limit=64) (float: 1.5)

    a = rUnit(3, 5)
    b = rUnit(7, 10)
    c = a & b  # Test bitwise AND operation
    print(f"{a} & {b} = {c}")

    d = rUnit(15, 1)
    e = 7
    f = d & e  # Test bitwise AND with integer
    print(f"{d} & {e} = {f}")

    a = rUnit(9, 4)
    b = rUnit(2, 1)

    mod_result = a % b
    print(f"{a} % {b} = {mod_result} (float: {mod_result.to_float()})")

    c = rUnit(5, 2)
    d = 1.5  # float
    mod_result2 = c % d
    print(f"{c} % {d} = {mod_result2} (float: {mod_result2.to_float()})")
    
    # Example 1: rUnit with positive value
    a = rUnit(9, 16, bit_limit=64)
    sqrt_a = a.sqrt()
    print(f"Square root of {a}: {sqrt_a} (float: {sqrt_a.to_float()})")

    # Example 2: Large rUnit
    b = rUnit(398589196000000000000000, 6871, bit_limit=512)
    sqrt_b = b.sqrt()
    print(f"Square root of {b}: {sqrt_b} (float: {sqrt_b.to_float()})")

    # Initialize test cases
    a = rUnit(3, 5, bit_limit=31)  # 0.6
    b = rUnit(7, 10, bit_limit=31)  # 0.7
    c = rUnit(15, 1, bit_limit=31)  # 15.0
    d = rUnit(7, 1, bit_limit=31)   # 7
    e = rUnit(9, 4, bit_limit=31)   # 2.25
    f = rUnit(2, 1, bit_limit=31)   # 2.0
    g = rUnit(5, 2, bit_limit=31)   # 2.5
    h = rUnit(3, 2, bit_limit=31)   # 1.5

    # Bitwise AND
    and_result1 = a & b
    and_result2 = c & d
    print(f"{a} & {b} = {and_result1}")
    print(f"{c} & {d} = {and_result2}")

    # Modulus
    mod_result1 = e % f
    mod_result2 = g % h
    print(f"{e} % {f} = {mod_result1} (float: {mod_result1.to_float()})")
    print(f"{g} % {h} = {mod_result2} (float: {mod_result2.to_float()})")

    # XOR
    xor_result1 = a ^ b
    xor_result2 = c ^ d
    print(f"{a} ^ {b} = {xor_result1}")
    print(f"{c} ^ {d} = {xor_result2}")


    a = rUnit(123456, 789012, bit_limit=32)
    b = rUnit(987654, 321098, bit_limit=32)
    result = ((a + b) * a) / b
    print(result.to_float())

    a = rUnit(2**15, 3, bit_limit=16)
    b = rUnit(1, 3, bit_limit=16)
    print(a + b)

    a = rUnit(15, 1, bit_limit=31)
    b = rUnit(7, 1, bit_limit=31)
    print(a & b)  # Expect 7/1
    print(a ^ b)  # Expect 8/1

    a = rUnit(10**12, 1, bit_limit=64)
    b = rUnit(1, 10**12, bit_limit=64)
    result = a * b
    print(result.to_float())  # Expect 1.0

    a = rUnit(2, 4)
    b = rUnit(1, 2)
    print(a == b)  # Expect True

    a = rUnit(25, 4)
    b = rUnit(5, 4)
    print(a % b)  # Expect 0/1

    a = rUnit(1, 0)  # Should handle gracefully
    b = rUnit(0, 1)
    print(a + b)  # Expect 0/1

    import random
    noise = random.uniform(-1e-6, 1e-6)
    a = rUnit(1, 3) + noise
    b = rUnit.from_float(1/3 + noise)
    print("a ", a)  # Expect True
    print("b ", b)  # Expect True
    print("noise ", noise)  # Expect True
    print(a == b)  # Expect True

    # Original rUnit
    a = rUnit(10288, 65751, bit_limit=32)

    # Round to two decimal places (precision=100)
    rounded_a_2dp = a.round_to_precision(100)
    print(f"Rounded to 0.01: {rounded_a_2dp} (float: {rounded_a_2dp.to_float()})")

    # Round to the nearest integer (precision=1)
    rounded_a_int = a.round_to_precision(1)
    print(f"Rounded to integer: {rounded_a_int} (float: {rounded_a_int.to_float()})")


    # Original rUnit
    a = rUnit(10288, 65751, bit_limit=32)

    # Default floating-point conversion
    print(f"Default to_float: {a.to_float()}")

    # Conversion with precision
    print(f"To float with precision (10): {a.to_float_with_precision(10)}")
    print(f"To float with precision (5): {a.to_float_with_precision(5)}")

    # Adding two rUnits
    a = rUnit(1, 2)
    b = rUnit(1, 3)
    print(f"{a} + {b} = {a + b} (float: {(a + b).to_float()})")

    # Adding an rUnit and an integer
    c = rUnit(2, 5)
    d = 1
    print(f"{c} + {d} = {c + d} (float: {(c + d).to_float()})")

    # Adding an rUnit and a float
    e = rUnit(3, 7)
    f = 0.25
    print(f"{e} + {f} = {e + f} (float: {(e + f).to_float()})")

    # Error handling
    try:
        g = rUnit(1, 3)
        h = "invalid"
        print(g + h)
    except TypeError as ex:
        print(f"Error: {ex}")

    test_rUnit_sum_series(100000)  # Test with n=10
   
    # Define the stages of the gearbox as (driving teeth, driven teeth)
    gear_stages = [
        (30, 90),  # Stage 1 1/3
        (45, 15),  # Stage 2 3/1
        (20, 40),  # Stage 3 1/2
    ]
    
    gear_stages = [
        (100, 50),  # Stage 1: 2:1
        (50, 10),   # Stage 2: 5:1
    ]

    gear_stages = [
        (50, 250),  # Stage 1: Reduction 5:1
    ]

   
    gear_stages = [
        (133, 25),  # Stage 1: 2:1
        (25, 1526),   # Stage 2: 5:1
    ]


    result = calculate_gear_ratio(gear_stages)



if __name__ == "__main__":
    selfTest()