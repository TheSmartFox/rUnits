
import math
from math import gcd,isqrt


class rUnit:
    def __init__(self, numerator, denominator=1, bit_limit=63, is_dynamic=0):
        """
        Initialize rUnit with a numerator and denominator.
        Gracefully handles zero denominator by setting the value to 0.
        """
        self.default_bit_limit = 63
        self.sign = 0 #positive
        self.bit_limit = bit_limit
        
        if numerator < 0:
            self.sign = 1
            # numerator = -1*numerator
        
        #if denominator < 0:
        #     denominator = -1*denominator

        self.is_dynamic = is_dynamic           
        if self.bit_limit == 0:
            self.is_dynamic = 1
            self.bit_limit = self.default_bit_limit
        else:
            self.bit_limit = bit_limit
        
        # if self.bit_limit == self.default_bit_limit:
        #     self.is_dynamic = 0   
        # else:
        #     self.is_dynamic = 1   
            
        
        self.numerator = 0
        self.denominator = 1
        
        while numerator%2==0 and denominator%2==0 and denominator != 0 and numerator != 0:
            numerator = numerator // 2
            denominator = denominator // 2
            
        while numerator%3==0 and denominator%3==0 and denominator != 0 and numerator != 0:
            numerator = numerator // 3
            denominator = denominator // 3
        
        while numerator%5==0 and denominator%5==0 and denominator != 0 and numerator != 0:
            numerator = numerator // 5
            denominator = denominator // 5
            
        if denominator == 0:
            # Set to zero when denominator is zero
            self.numerator = 0
            self.denominator = 1
        else:
            # Apply the bit limit and simplify
            numerator, denominator = self._apply_bit_limit(numerator, denominator)
            common = gcd(self.numerator, self.denominator)
            self.numerator = numerator // common
            self.denominator = denominator // common


    def adjust_bit_limit(self, numerator, denominator):
        if self.is_dynamic == 1:
        
            max_bits = max(numerator.bit_length(), denominator.bit_length())
            max_bits = max(max_bits, self.default_bit_limit)
            #print("max bits " + str(max_bits))
            if max_bits > self.default_bit_limit:
                is_dynamic_limit = ((max_bits + (self.default_bit_limit-1)) // self.default_bit_limit) * self.default_bit_limit  # Round up to nearest 32-bit chunk
                return is_dynamic_limit
            return self.default_bit_limit
        
        else:
            return self.bit_limit
            

    def _apply_bit_limit(self, numerator, denominator):
        """
        Apply bit limit by truncating numerator and denominator.
        Handles cases where numerator or denominator might be rUnit instances.
        """
        # Ensure numerator and denominator are primitive types
        # if isinstance(numerator, rUnit):
        #     numerator = numerator.to_float()
        # if isinstance(denominator, rUnit):
        #     denominator = denominator.to_float()

        numerator = int(numerator)
        denominator = int(denominator)

        self.bit_limit = self.adjust_bit_limit(numerator, denominator)

        # Define the maximum value based on the bit limit
        max_value = (1 << self.bit_limit) - 1

        # Apply bit limit using clamping
        numerator = max(0, min(numerator, max_value))
        denominator = max(1, min(denominator, max_value))  # Avoid zero denominator

        return numerator, denominator


    def simplify(self):
        """
        Simplify the fraction by dividing numerator and denominator by their GCD.
        """
        
        while numerator%2==0 and denominator%2==0 and denominator != 0 and numerator != 0:
            numerator = numerator // 2
            denominator = denominator // 2
            
        while numerator%3==0 and denominator%3==0 and denominator != 0 and numerator != 0:
            numerator = numerator // 3
            denominator = denominator // 3
        
        while numerator%5==0 and denominator%5==0 and denominator != 0 and numerator != 0:
            numerator = numerator // 5
            denominator = denominator // 5


        if self.denominator == 0:
            self.numerator = 0
            self.denominator = 1
        else:
            common = gcd(self.numerator, self.denominator)
            self.numerator //= common
            self.denominator //= common

    def to_float(self):
        """
        Convert rUnit to a floating-point value.
        """
        #self.simplify()
        if self.denominator == 0:
            return 0.0
        return self.numerator / self.denominator

    def __add__(self, other):
        """
        Add two rUnit instances or an rUnit with a numeric value.
        """
        if not isinstance(other, rUnit):
            if isinstance(other, (int, float)):
                other = rUnit.from_float(other, bit_limit=self.bit_limit)
            else:
                raise TypeError(f"Unsupported operand type(s) for +: 'rUnit' and '{type(other).__name__}'")

        # Cross-multiply to find the new numerator and denominator
        num = (self.numerator * other.denominator) + (other.numerator * self.denominator)
        den = self.denominator * other.denominator

        # Apply the bit limit and simplify the result
        #num, den = self._apply_bit_limit(num, den)
        result = rUnit(num, den, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        #result.simplify()  # Ensure the result is in simplest form

        return result

    def __radd__(self, other):
        """
        Handle addition when the left operand is not an rUnit.
        """
        return self + other  # Reuse __add__ logic

    def __sub__(self, other):
        if not isinstance(other, rUnit):
            other = rUnit(other, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        num = (self.numerator * other.denominator) - (other.numerator * self.denominator)
        den = self.denominator * other.denominator
        #num, den = self._apply_bit_limit(num, den)
        result = rUnit(num, den, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        #result.simplify()  # Ensure the result is in simplest form

        return result

    def __xor__(self, other):
        """
        Compute the bitwise XOR operation.
        """
        if not isinstance(other, rUnit):
            other = rUnit(other, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)

        # Convert to integers within the bit limit
        max_value = (1 << self.bit_limit) - 1
        numerator_xor = (self.numerator & max_value) ^ (other.numerator & max_value)
        denominator_xor = (self.denominator & max_value) ^ (other.denominator & max_value)

        result = rUnit(numerator_xor, denominator_xor, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        #result.simplify()  # Ensure the result is in simplest form

        return result
    
    def __mul__(self, other):
        """
        Multiply two rUnits, or an rUnit with a numeric value.
        """
        if isinstance(other, rUnit):
            new_numer = self.numerator * other.numerator
            new_denom = self.denominator * other.denominator
            return rUnit(new_numer, new_denom, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        elif isinstance(other, int):
            # Multiply only the numerator for integer scaling
            new_numer = self.numerator * other
            new_denom = self.denominator
            return rUnit(new_numer, new_denom, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        elif isinstance(other, float):
            # Convert float to rUnit and multiply
            return self * rUnit.from_float(other, bit_limit=self.bit_limit)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'rUnit' and '{type(other).__name__}'")

    def __rmul__(self, other):
        """
        Support reversed multiplication.
        """
        return self * other

    
    def __mod__(self, other):
        """
        Compute the modulus operation directly on numerator and denominator.
        """
        if not isinstance(other, rUnit):
            other = rUnit(other, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)

        if other.numerator == 0:
            raise ZeroDivisionError("Modulo by zero is undefined.")

        # Align the fractions to the same denominator
        num_self = self.numerator * other.denominator
        num_other = other.numerator * self.denominator
        denom = self.denominator * other.denominator

        # Cast to integers explicitly to avoid type issues
        num_self = int(num_self)
        num_other = int(num_other)
        denom = int(denom)

        # Compute the modulus on the aligned numerators
        mod_numerator = num_self % num_other

        # Return the result as a simplified rUnit
        return rUnit(mod_numerator, denom, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)


    def __truediv__(self, other):
        if not isinstance(other, rUnit):
            other = rUnit(other, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        if other.numerator == 0:
            # Handle division by zero
            return rUnit(0, 1, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        num = self.numerator * other.denominator
        den = self.denominator * other.numerator
        #num, den = self._apply_bit_limit(num, den)
        return rUnit(num, den, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)

    def __repr__(self):
        if self.is_dynamic == 1:
            return f"{self.numerator}/{self.denominator} (bit_limit={self.bit_limit} [dynamic])"        
        else:
            return f"{self.numerator}/{self.denominator} (bit_limit={self.bit_limit})"

    # def __repr__(self):
    #     return f"{self.numerator}/{self.denominator} (bit_limit={self.bit_limit}) (float: {self.to_float()})"

    def __eq__(self, other):
        if not isinstance(other, rUnit):
            other = rUnit(other, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        return self.numerator * other.denominator == self.denominator * other.numerator

    def __neg__(self):
        return rUnit(-self.numerator, self.denominator, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
    
    def __and__(self, other):
            """
            Perform a bitwise AND operation.
            """
            if isinstance(other, rUnit):
                numerator = self.numerator & other.numerator
                denominator = self.denominator & other.denominator
                return rUnit(numerator, denominator, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
            elif isinstance(other, int):
                numerator = self.numerator & other
                denominator = self.denominator
                return rUnit(numerator, denominator, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
            else:
                raise TypeError(f"Unsupported operand type(s) for &: 'rUnit' and '{type(other).__name__}'")
            
    def __floordiv__(self, other):
        """
        Perform floor division (//) for rUnit.
        """
        if not isinstance(other, rUnit):
            other = rUnit(other, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        if other.numerator == 0:
            raise ZeroDivisionError("Floor division by zero is undefined.")
        
        # Perform floor division
        result_numerator = (self.numerator * other.denominator) // (self.denominator * other.numerator)
        return rUnit(result_numerator, 1, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)


    def exp(self):
        """
        Calculate the exponential of the rUnit number.
        Uses Python's math.exp with overflow handling.
        """
        value = self.to_float()
        if value > 709:  # Prevent overflow
            return rUnit(2**self.bit_limit - 1, 1, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)  # Approximation of infinity in rUnit
        elif value < -709:  # Prevent underflow
            return rUnit(0, 1, self.bit_limit)  # Approximation of zero in rUnit
        else:
            return rUnit.from_float(math.exp(value), bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)

    def tanh(self, x):
        """
        Compute the tanh activation using rUnit.
        """
        scaled_x = x / rUnit(10, 1, self.bit_limit)  # Scale down the input
        exp_pos = (scaled_x * rUnit(2, 1, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)).exp()
        exp_neg = (scaled_x * rUnit(-2, 1, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)).exp()
        return (exp_pos - exp_neg) / (exp_pos + exp_neg)

    def round_to_precision(self, precision: int) -> 'rUnit':
        """
        Round the rUnit value to the nearest multiple of 1/precision.
        """
        if precision <= 0:
            raise ValueError("Precision must be a positive integer.")

        # Scale the numerator to match the precision and round
        scale_factor = precision * self.denominator
        rounded_numerator = round(self.numerator * precision / self.denominator)
        
        # Return a new rUnit representing the rounded value
        return rUnit(rounded_numerator, scale_factor, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)

    def to_float_with_precision(self, precision: int = 15) -> float:
        """
        Convert rUnit to a floating-point value with a specified precision.
        Default precision is 15 decimal places.
        """
        if self.denominator == 0:
            return 0.0  # Avoid division by zero, treated as 0
        # Adjust numerator and denominator to the precision level
        scale_factor = 10**precision
        precise_numerator = round(self.numerator * scale_factor / self.denominator)
        return precise_numerator / scale_factor

    def sqrt(self):
        """
        Compute the square root of the rUnit.
        The square root is computed directly using integer arithmetic to maintain precision.
        """
        if self.numerator < 0:
            raise ValueError("Cannot compute the square root of a negative rUnit.")
        
        # Integer square root for numerator and denominator
        sqrt_numerator = isqrt(self.numerator)
        sqrt_denominator = isqrt(self.denominator)

        # Simplify the result to respect the bit limit
        return rUnit(sqrt_numerator, sqrt_denominator, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
    
    def pow(self, exponent):
        """
        Compute the power of the current rUnit raised to an integer exponent.
        """
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer.")
        if exponent < 0:
            # Handle negative exponents
            return rUnit(self.denominator**-exponent, self.numerator**-exponent, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        else:
            return rUnit(self.numerator**exponent, self.denominator**exponent, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)

    def modular_root(a, q, mod):
        """
        Compute modular root: x^q ≡ a (mod n)
        Only works for specific cases and small values of q.
        """
        for x in range(mod):
            if pow(x, q, mod) == a:
                return x
        return None  # No solution

    def mod_pow(self, exponent, modulus):
        """
        Compute modular exponentiation for rUnits: (self^exponent) % modulus.
        Supports integer and fractional exponents.
        """
        if not isinstance(modulus, rUnit):
            modulus = rUnit(modulus, 1, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)
        if not isinstance(exponent, rUnit):
            exponent = rUnit(exponent, 1, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)

        # Handle fractional exponents
        if exponent.denominator != 1:
            # Fractional exponent: a^(p/q) mod n
            p = exponent.numerator
            q = exponent.denominator
            if p < 0 or q < 0:
                raise ValueError("Negative exponents are not supported in modular arithmetic.")

            # Compute a^p mod n
            base = self.numerator
            mod = modulus.numerator
            mod_pow_result = pow(base, p, mod)  # Compute a^p mod n

            # Attempt modular root
            root_mod = self.modular_root(mod_pow_result, q, mod)
            if root_mod is None:
                raise ValueError(f"Cannot compute modular root for q={q} under modulus={mod}")
            return rUnit(root_mod, 1, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)

        # Integer exponent case
        base = self.numerator
        exp = exponent.numerator
        mod = modulus.numerator
        if mod == 0:
            raise ZeroDivisionError("Modulo by zero is undefined.")

        result = pow(base, exp, mod)
        return rUnit(result, 1, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)

    def mod_inverse(self, modulus):
        """
        Calculate the modular inverse of an rUnit under the given modulus.
        Uses the Extended Euclidean Algorithm.
        """
        if not isinstance(modulus, rUnit):
            modulus = rUnit(modulus, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)

        if modulus.numerator <= 1:
            raise ValueError("Modular inverse is not defined for modulus <= 1.")
        
        if self.numerator == 0:
            raise ValueError("Modular inverse does not exist for 0.")

        a = self.numerator % modulus.numerator
        m = modulus.numerator

        # Extended Euclidean Algorithm
        m0, x0, x1 = m, 0, 1
        while a > 1:
            # Quotient
            q = a // m
            a, m = m, a % m

            # Update x0 and x1
            x0, x1 = x1 - q * x0, x0

        # Make x1 positive
        if x1 < 0:
            x1 += m0

        return rUnit(x1, 1, bit_limit=self.bit_limit, is_dynamic=self.is_dynamic)


        # Serialized format (dynamic example):
        # [1 bit sign][1 dynamic][if dynamic Metadata (16 bits)][Numerator (64 bits)][Denominator (64 bits)]

    def serialize(self):
        # Prepare sign and dynamic flag
        sign_bit = 0 if self.numerator >= 0 else 1
        dynamic_flag = 1 if self.is_dynamic else 0
        header = (sign_bit << 1) | dynamic_flag

        if not self.is_dynamic:
            # Static: fixed 64-bit representation
            numerator = abs(self.numerator) & ((1 << self.default_bit_limit) - 1)
            denominator = self.denominator & ((1 << self.default_bit_limit) - 1)
            return [header, numerator, denominator]
        else:
            # Dynamic: include metadata
            numerator_bits = self.numerator.bit_length()
            denominator_bits = self.denominator.bit_length()
            metadata = (numerator_bits << 16) | denominator_bits
            return [header, metadata, self.numerator, self.denominator]

    def deserialize(self, data):
        header = data[0]
        sign_bit = (header >> 1) & 1
        dynamic_flag = header & 1

        if not dynamic_flag:
            # Static: fixed 64-bit representation
            numerator = data[1]
            denominator = data[2]
        else:
            # Dynamic: use metadata
            metadata = data[1]
            numerator_bits = (metadata >> 16) & 0xFF
            denominator_bits = metadata & 0xFF
            numerator = data[2]
            denominator = data[3]

        # Apply sign and return reconstructed rUnit
        numerator = -numerator if sign_bit else numerator
        return rUnit(numerator, denominator, bit_limit=self.default_bit_limit if not dynamic_flag else max(numerator_bits, denominator_bits))


    @staticmethod
    def from_float(value, bit_limit=64):
        """
        Create an rUnit from a float value.
        """
        denominator = 10**15  # Define precision level
        numerator = int(value * denominator)
        return rUnit(numerator, denominator, bit_limit)

