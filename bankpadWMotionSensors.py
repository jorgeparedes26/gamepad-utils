#!/usr/bin/env python3
"""
bankpadWMotionSensors.py - merge TWO /dev/input/event so we have ONE virtual pad with combined input
from axis and gyro for pitch, yaw and a simulated axis for banking, for dxx-redux/rebirth.

Usage:
    sudo python3 bankpadWMotionSensors.py <padA-event> <padA-motionEvent> [virtual-name]
    <padA-event>   e.g. /dev/input/event24  (Gamepad: full controls + throttle)
    <padA-motionEvent>  /dev/input/event25  (Motion sensors from gamepad)
    [virtual-name] optional; defaults to "DS4 bankpadWMotionSensors"

Ctrl-C to stop; the virtual pad disappears on exit.
"""

import sys
import select
import evdev
import math
from evdev import ecodes, AbsInfo, UInput

# ---- tunables -------------------------------------------------------------
DEFAULT_NAME = "DS4 bankpadWMotionSensors"
TRIGGER_MAX  = 255
INNER_DEADZONE     = 4
AXIS_OUT_MIN      = -128
AXIS_OUT_MAX      =  127
GYRO_BANK_AXIS     = ecodes.ABS_X   # pad A gyro z axis -> bank

BANK_AXIS     = ecodes.ABS_MISC   # pad A triggers -> bank
THROTTLE_AXIS = ecodes.ABS_Z      # pad B triggers -> throttle

BANK_INVERT = True   # bank = R2 - L2  
THROTTLE_INVERT = False   # throttle (R2 accel +, L2 reverse -)
PITCH_INVERT = True
VERTICAL_SLIDE_INVERT = True

BANK_SENSITIVITY = 1.2
SNIPING_SENSITIVITY_FACTOR = 0.35         # Reduces aiming speed when sniper_mode_active == True

L2, R2 = ecodes.ABS_Z, ecodes.ABS_RZ   # raw trigger source codes on each pad

# ---- button layers ---------------------------------------------------------

# ---- button layers --------------------------------------------------------
# Two modifiers create 16 extra virtual gamepad buttons. A third modifier
# (D-pad Down) exposes keyboard keys so game/menu navigation can be done
# without leaving the gamepad.
MOD_SELECT = ecodes.BTN_SELECT
MOD_DPAD_UP_AXIS = ecodes.ABS_HAT0Y
MOD_DPAD_DOWN_AXIS = ecodes.ABS_HAT0Y


# Nine buttons that can acquire alternate meanings.
# Special case: BTN_TL2 is MANDATED for Afterburner, a la PS1 Version???
LAYER_BUTTONS = (
    ecodes.BTN_SOUTH,   # Cross
    ecodes.BTN_EAST,    # Circle
    ecodes.BTN_NORTH,   # Triangle
    ecodes.BTN_WEST,    # Square
    ecodes.BTN_TL,      # L1
    ecodes.BTN_TR,      # R1
    ecodes.BTN_THUMBL,  # L3
    ecodes.BTN_THUMBR,  # R3
    ecodes.BTN_TL2      #L2 
)   

# Exactly 27 extra virtual buttons.
EXTRA_BUTTONS = tuple(
    getattr(ecodes, f"BTN_TRIGGER_HAPPY{i}")
    for i in range(1, 28) 
)

# State variables
state = {
    'sniper_mode_active': False,
    'select_held': False,
    'dpad_up_held': False,
    'dpad_down_held': False,
    # physical source button -> (event type, output code)
    # This is what makes button releases safe if a modifier is released first.
    'active_layered_buttons': {}
}

SNIPER_MODE_BUTTON = ecodes.ABS_HAT0X 
# ---------------------------------------------------------------------------

def scale_trigger(v):
    if abs(v) <= INNER_DEADZONE:
        return 0.0
    span = TRIGGER_MAX - INNER_DEADZONE
    return max(0.0, min(1.0, (v - INNER_DEADZONE) / span)) if span > 0 else 0.0

def merge(a, b, invert):
    """Return centered -OUT..+OUT from two 0..1 trigger magnitudes."""
    val = (a - b) if invert else (b - a)   # default: R2(b) - L2(a)
    val = max(-1.0, min(1.0, float(val)))
    if val >= 0:
        return int(min(val * AXIS_OUT_MAX, AXIS_OUT_MAX))
    return int(max(val * abs(AXIS_OUT_MIN), AXIS_OUT_MIN))

# Convert 0..255 raw value to -32768..32767
# def scale_axis_255_to_32767(raw_val):
#     # Center the 0..255 range around 0 (-128 to 127)
#     centered = raw_val - 128
    
#     # Scale up to 16-bit signed integer limits
#     if centered < 0:
#         return int((centered / 128.0) * 32768)
#     else:
#         return int((centered / 127.0) * 32767)

# def normalize_joystick_axis_from_16bits(raw_val: int) -> float:
#     """Converts raw signed 16-bit joystick values (-32768 to 32767) cleanly to -1.0..1.0."""
#     raw_val = max(-32768, min(32767, raw_val))
#     if raw_val < 0:
#         return raw_val / 32768.0
#     return raw_val / 32767.0

def normalize_joystick_axis_from_8bits(raw_val: int) -> float:
    """Converts a range (-128 to 127) cleanly to -1.0..1.0."""
    centered = raw_val - (-AXIS_OUT_MIN)
    
    raw_val = max(AXIS_OUT_MIN, min(AXIS_OUT_MAX, centered))
    if raw_val < 0:
        return raw_val / - AXIS_OUT_MIN
    return raw_val / AXIS_OUT_MAX

# def denormalize_joystick_axis(normalized_val: float) -> int:
#     """Converts a processed -1.0..1.0 float back to a signed 16-bit integer."""
#     normalized_val = max(-1.0, min(1.0, normalized_val))
#     if normalized_val < 0:
#         return round(normalized_val * 32768)
#     return round(normalized_val * 32767)

def denormalize_joystick_axis_to_8bits(normalized_val: float) -> int:
    """Converts a processed -1.0..1.0 float back to a signed 16-bit integer."""
    normalized_val = max(-1.0, min(1.0, normalized_val))
    if normalized_val < 0:
        return round(normalized_val * -AXIS_OUT_MIN)
    return round(normalized_val * AXIS_OUT_MAX)

def radial_to_square_translation_sliding(x: float, y: float,
                                        return_axis_x: bool = True,
                                        return_axis_y: bool = True   ) -> tuple[float, float]:
    """
    Stretches a circular joystick profile into a square profile.
    This guarantees that pushing a DualShock 4 into diagonal edges achieves 1.0, 1.0
    allowing full-velocity trichording in Descent ports.
    """
    if abs(x) < 1e-5 and abs(y) < 1e-5:
        return 0.0, 0.0
        
    # Determine the angle of deflection
    phi = math.atan2(y, x)
    abs_cos_phi = abs(math.cos(phi))
    abs_sin_phi = abs(math.sin(phi))
    
    # Calculate scaling factor to project circle boundary out to the square wall
    if abs_cos_phi >= abs_sin_phi:
        scale = 1.0 / abs_cos_phi
    else:
        scale = 1.0 / abs_sin_phi
        
    # Scale components out smoothly
    return max(-1.0, min(1.0, x * scale)) if return_axis_x else 0.0, max(-1.0, min(1.0, y * scale)) if return_axis_y else 0

# =====================================================================
# 1. S-CURVE METHODS (Flattened at center AND flattened at full tilt)
# =====================================================================

def scurve_cubic(x: float, a: float = 0.5) -> float:
    """
    Applies a Cubic Polynomial S-Curve.
    
    Args:
        x (float): Raw joystick input, normalized from -1.0 to 1.0.
        a (float): Tweak parameter from 0.0 (pure linear) to 1.0 (pure cubic).
        
    Returns:
        float: Modified output from -1.0 to 1.0.
    """
    # Clamp input to avoid out-of-bounds math errors
    x = max(-1.0, min(1.0, x))
    
    # Formula: y = a*x^3 + (1-a)*x
    # Preserves sign naturally because x^3 of a negative is negative
    return a * (x ** 3) + (1.0 - a) * x


def scurve_smoothstep(x: float, a: float = 1.0) -> float:
    """
    Applies a Smoothstep (Hermite Interpolation) S-Curve.
    
    Args:
        x (float): Raw joystick input, normalized from -1.0 to 1.0.
        a (float): Blending factor from 0.0 (pure linear) to 1.0 (pure smoothstep).
        
    Returns:
        float: Modified output from -1.0 to 1.0.
    """
    x = max(-1.0, min(1.0, x))
    abs_x = abs(x)
    
    # Standard Smoothstep formula for [0.0, 1.0] domain: 3x^2 - 2x^3
    smooth_x = 3 * (abs_x ** 2) - 2 * (abs_x ** 3)
    
    # Blend with linear input based on the 'a' parameter
    result = a * smooth_x + (1.0 - a) * abs_x
    
    # Restore the original direction (positive/negative)
    return math.copysign(result, x)


def scurve_trigonometric(x: float, a: float = 1.0) -> float:
    """
    Applies a Sine-Based Trigonometric S-Curve for an organic feel.
    
    Args:
        x (float): Raw joystick input, normalized from -1.0 to 1.0.
        a (float): Blending factor from 0.0 (pure linear) to 1.0 (pure sine curve).
        
    Returns:
        float: Modified output from -1.0 to 1.0.
    """
    x = max(-1.0, min(1.0, x))
    abs_x = abs(x)
    
    # Standard sine curve mapped to push [0, 1] input smoothly into a [0, 1] output S-shape
    sine_x = 0.5 * math.sin(math.pi * (abs_x - 0.5)) + 0.5
    
    # Blend with linear input
    result = a * sine_x + (1.0 - a) * abs_x
    
    return math.copysign(result, x)


def scurve_sigmoid(x: float, k: float = 6.0) -> float:
    """
    Applies a Logistic Sigmoid S-Curve. 
    Includes normalization to guarantee hitting exactly (0,0) and (1,1).
    
    Args:
        x (float): Raw joystick input, normalized from -1.0 to 1.0.
        k (float): Steepness factor. Higher = flatter center, steeper drop-off. 
                   Recommended range: 2.0 to 10.0. (0.0 approaches linear).
                   
    Returns:
        float: Modified output from -1.0 to 1.0.
    """
    x = max(-1.0, min(1.0, x))
    abs_x = abs(x)
    
    # Handle the edge case where k is 0 to avoid zero-division errors in normalization
    if abs(k) < 1e-5:
        return x
        
    # Standard sigmoid formula centered at x = 0.5
    raw_sigmoid = lambda val: 1.0 / (1.0 + math.exp(-k * (val - 0.5)))
    
    # Normalization boundaries because a standard sigmoid never truly touches 0 or 1
    sig_at_0 = raw_sigmoid(0.0)
    sig_at_1 = raw_sigmoid(1.0)
    
    # Scale and shift the sigmoid output so 0.0 maps to 0.0 and 1.0 maps to 1.0
    normalized_sig = (raw_sigmoid(abs_x) - sig_at_0) / (sig_at_1 - sig_at_0)
    
    return math.copysign(normalized_sig, x)


# =====================================================================
# 2. EXPONENTIAL METHODS (Flattened at center, highly aggressive at edge)
# =====================================================================

def expo_pure_jcurve(x: float, a: float = 2.0) -> float:
    """
    Applies a Pure Exponential Curve (J-Curve).
    
    Args:
        x (float): Raw joystick input, normalized from -1.0 to 1.0.
        a (float): Exponent factor. 1.0 is linear. >1.0 expands the center deadzone.
                   Recommended range: 1.5 to 4.0.
                   
    Returns:
        float: Modified output from -1.0 to 1.0.
    """
    x = max(-1.0, min(1.0, x))
    
    # Formula: y = sign(x) * |x|^a
    # Extracts absolute value, applies power, restores direction sign
    return math.copysign(abs(x) ** a, x)


def expo_multiplier_blend(x: float, expo: float = 0.5) -> float:
    """
    Applies a blended exponential approximation common in flight/RC software.
    
    Args:
        x (float): Raw joystick input, normalized from -1.0 to 1.0.
        expo (float): Curve blend parameter from 0.0 (pure cubic) to 1.0 (pure linear).
        
    Returns:
        float: Modified output from -1.0 to 1.0.
    """
    x = max(-1.0, min(1.0, x))
    
    # Formula: y = x * expo + x^3 * (1 - expo)
    # Blends a standard linear line with a cubic response curve
    return x * expo + (x ** 3) * (1.0 - expo)

#Apply a clean 2D S-Curve specifically to your Right Stick
def s_2d_curve(x: float) -> float:
    # Apply an elegant Smoothstep curve for aiming precision near center
    curved_mag = 3 * (x ** 2) - 2 * (x ** 3)
    return curved_mag


def apply_deadzone(
    x: float, 
    lower_deadzone: float = 0.05, 
    upper_deadzone: float = 1.0
) -> float:
    """
    Applies lower deadzone (drift protection) and undercalibration/upper deadzone
    (maximum threshold trimming) to a normalized joystick input, ensuring a 
    smooth, linear scale between the thresholds.
    
    Args:
        x (float): Raw joystick value directly from the hardware, from -1.0 to 1.0.
        lower_deadzone (float): Inner dead space percentage (e.g., 0.05 targets 5%).
                                Inputs inside this range return exactly 0.0.
        upper_deadzone (float): Outer ceiling point for undercalibration (e.g., 0.90 targets 90%).
                                Inputs outside this value return maximum output (1.0).
                                
    Returns:
        float: A perfectly re-scaled linear value bounded strictly from -1.0 to 1.0.
    """
    # 1. Hardware Guard: Clamp input immediately to preserve hardware limitations
    x = max(-1.0, min(1.0, x))
    
    # Extract absolute magnitude for mapping calculations; preserve sign for vector output
    abs_x = abs(x)
    
    # 2. Check Lower Bound: Is input wandering inside the inner stick-drift region?
    if abs_x < lower_deadzone:
        return 0.0
    # 3. Check Upper Bound (Undercalibration): Has the stick maxed out early?
    elif abs_x > upper_deadzone:
        return math.copysign(1.0, x)
            
    # 4. Range Rescaling: Linear interpolation across the active sweet-spot window.
    # Scale shifting prevents a huge "jump" or step-up immediately when moving out of center.
    # Range is defined as the active delta between the upper ceiling and lower floor.
    active_range = upper_deadzone - lower_deadzone
    
    # If parameters match or conflict, default safely to avoiding division by zero
    if active_range <= 1e-5:
        return math.copysign(1.0, x)
        
    # Standard normalization formula: (Value - NewFloor) / (NewCeiling - NewFloor)
    rescaled_value = (abs_x - lower_deadzone) / active_range
    
    # Re-apply the vector heading direction (positive or negative tilt)
    return math.copysign(rescaled_value, x)

def calibrate_axis(raw_value: float, actual_max: float, target_min: float = 0.0, target_max: float = 1.0) -> float:
    """
    Rescales a restricted joystick axis value to express the full intended range.
    
    Use this when a worn-out or uncalibrated joystick physically cannot reach 
    its maximum output value, causing it to max out early in software.
    
    Parameters:
    -----------
    raw_value : float
        The current raw input reading from the joystick axis.
    actual_max : float
        The maximum raw value the joystick can actually express (e.g., 0.85 instead of 1.0).
    target_min : float, optional
        The desired minimum output value. Defaults to 0.0.
    target_max : float, optional
        The desired maximum output value. Defaults to 1.0.
        
    Returns:
    --------
    float
        The calibrated and scaled axis value, clamped safely between target_min and target_max.
    """
    # 1. Clamp the raw value to ensure it doesn't exceed the actual hardware maximum
    #    or drop below the absolute minimum (assumed 0.0 for a single-direction axis).
    clamped_value = max(0.0, min(raw_value, actual_max))
    
    # 2. Calculate how far along the clamped value is in the hardware's restricted range (0 to actual_max).
    #    This creates a ratio from 0.0 to 1.0.
    normalized_ratio = clamped_value / actual_max
    
    # 3. Map that 0.0 - 1.0 ratio to the target output range (target_min to target_max).
    calibrated_value = target_min + (normalized_ratio * (target_max - target_min))
    
    return calibrated_value

def calibrate_asymmetric_axis(raw_value: float, actual_min: float, actual_max: float) -> float:
    """
    Calibrates a bidirectional joystick axis where the maximum positive and 
    negative limits are restricted non-symmetrically.
    
    Parameters:
    -----------
    raw_value : float
        The current raw input reading from the joystick axis (centered near 0.0).
    actual_min : float
        The maximum negative value the joystick can actually reach (e.g., -0.70).
        Must be a negative number.
    actual_max : float
        The maximum positive value the joystick can actually reach (e.g., 0.85).
        Must be a positive number.
        
    Returns:
    --------
    float
        The calibrated value scaled to a perfect -1.0 to 1.0 range.
    """
    # 1. Clamp the raw value safely within the known hardware limits
    clamped_value = max(actual_min, min(raw_value, actual_max))
    
    # 2. Determine if the joystick is pushed in the positive or negative direction
    if clamped_value >= 0.0:
        # Scale the positive half-range (0.0 to actual_max) into (0.0 to 1.0)
        return clamped_value / actual_max
    else:
        # Scale the negative half-range (actual_min to 0.0) into (-1.0 to 0.0)
        # Note: Dividing by the absolute value ensures the sign remains negative
        return clamped_value / abs(actual_min)

def calibrate_asymmetric_axis_with_deadzone(
    raw_value: float, 
    actual_min: float, 
    actual_max: float, 
    deadzone: float = 0.05
) -> float:
    """
    Calibrates an asymmetric bidirectional joystick axis and applies a deadzone.
    
    Parameters:
    -----------
    raw_value : float
        The current raw input reading from the joystick axis.
    actual_min : float
        The maximum physical negative value reached (e.g., -0.70). Must be negative.
    actual_max : float
        The maximum physical positive value reached (e.g., 0.85). Must be positive.
    deadzone : float, optional
        The threshold around 0.0 where input is ignored to prevent stick drift.
        Defaults to 0.05 (5% of the range).
        
    Returns:
    --------
    float
        The calibrated value smoothly scaled to a perfect -1.0 to 1.0 range.
    """
    # 1. Clamp the raw value safely within the broken hardware limits
    clamped_value = max(actual_min, min(raw_value, actual_max))
    
    # 2. Check if the value falls inside the central deadzone
    if abs(clamped_value) <= deadzone:
        return 0.0
        
    # 3. Process the positive direction
    if clamped_value > 0.0:
        # Scale the remaining active range (deadzone to actual_max) smoothly to (0.0 to 1.0)
        # This prevents the input from "snapping" from 0.0 straight to the deadzone value
        active_range = actual_max - deadzone
        if active_range <= 0:
            return 0.0
        return (clamped_value - deadzone) / active_range
        
    # 4. Process the negative direction
    else:
        # Scale the remaining active negative range (actual_min to -deadzone) smoothly to (-1.0 to 0.0)
        active_range = abs(actual_min) - deadzone
        if active_range <= 0:
            return 0.0
        return (clamped_value + deadzone) / active_range


def apply_radial_calibration_pitch_yaw(
    pitch: float, 
    yaw: float, 
    lower_deadzone: float = 0.1, 
    upper_deadzone: float = 0.7,
    sensitivity: float = 1.0,
    return_pitch_axis: bool = True,
    return_yaw_axis: bool = True    
) -> tuple[float, float]:

    # 1. Clamp inputs to ensure they don't break logic on hyper-deflected controllers
    pitch = max(-1.0, min(1.0, pitch))
    yaw = max(-1.0, min(1.0, yaw))
    
    # 2. Calculate the magnitude (hypotenuse) of the joystick vector using Pythagorean theorem
    magnitude = math.sqrt(pitch**2 + yaw**2)
    
    # Handle the perfect center edge case to prevent division by zero later
    if magnitude < 1e-5:
        return 0.0, 0.0
        
    # 3. Inner Radial Deadzone Check
    if magnitude <= lower_deadzone:
        return 0.0, 0.0
        
    # 4. Outer Deadzone / Undercalibration Check
    # If vector magnitude exceeds upper limit, push the vector completely to the edge
    if magnitude >= upper_deadzone:
        # Normalize the vector components and scale to maximum outer circle (1.0)
        # This preserves the user's intended angle perfectly
        scale = 1.0 / magnitude
        return pitch * scale, yaw * scale
        
    # 5. Radial Rescaling
    # Linearly map the magnitude between the inner deadzone and outer ceiling
    # active_range = upper_deadzone - lower_deadzone
    # if active_range <= 1e-5:
    #     scale = 1.0 / magnitude
    #     return x * scale, y * scale

    #Re-scale vector smoothly past the deadzone threshold
    active_range = upper_deadzone - lower_deadzone
    rescaled_magnitude = (magnitude - lower_deadzone) / active_range

    if state['sniper_mode_active']:
        scale = 1
    else:
        # Apply an elegant Smoothstep curve for aiming precision near center
        curved_mag = s_2d_curve(rescaled_magnitude)
        scale = curved_mag / magnitude
 
    # 6. Reconstruct the Vector components
    # Multiply the directional unit vector by our new calculated scale
    scale = rescaled_magnitude / magnitude
    pitch_out = yaw_out = 0
    
    if return_pitch_axis:
        pitch_out = pitch * scale * sensitivity
    if return_yaw_axis:
        yaw_out = yaw * scale * sensitivity

    return pitch_out, yaw_out


### =====================================================================

### VIRTUAL DEVICE CAPABILITIES

### =====================================================================
def build_caps_padA(srcA):
    """Clone pad A's caps, minus its raw triggers, plus BANK and THROTTLE axes.
    (THROTTLE lives here too because the virtual device is built from pad A.)"""
    caps = srcA.capabilities()
    out = {}
    for etype, entries in caps.items():
        if etype in (ecodes.EV_SYN, ecodes.EV_FF, ecodes.EV_FF_STATUS):
            continue
        
        if etype == ecodes.EV_ABS:
            axes = []
            present = set()
            for code, ai in entries:
                axes.append((code, AbsInfo(value=0, min=AXIS_OUT_MIN, max=AXIS_OUT_MAX,
                                                                fuzz=0, flat=0, resolution=0)))
                present.add(code)
            # for extra in (BANK_AXIS, THROTTLE_AXIS):
            #     if extra not in present:
            #         axes.append((extra, AbsInfo(value=0, min=AXIS_OUT_MIN, max=AXIS_OUT_MAX,
            #                                     fuzz=0, flat=0, resolution=0)))
            if BANK_AXIS not in present:
                axes.append((BANK_AXIS, AbsInfo(value=0, min=AXIS_OUT_MIN, max=AXIS_OUT_MAX,
                                                fuzz=0, flat=0, resolution=0)))
            if THROTTLE_AXIS not in present:
                axes.append((THROTTLE_AXIS, AbsInfo(value=0, min=AXIS_OUT_MIN, max=AXIS_OUT_MAX,
                                                fuzz=0, flat=0, resolution=0)))

            print("AXESSS")
            for axis in axes:
                print(axis)        
            out[ecodes.EV_ABS] = axes

        elif etype == ecodes.EV_KEY:
            keys = list(entries)

            for extra in EXTRA_BUTTONS:
                if extra not in keys:
                    keys.append(extra)

            out[ecodes.EV_KEY] = keys            
        else:
            out[etype] = list(entries)

    return out

def get_button_layer():
    """
    Return:
        0 = normal
        1 = Select modifier layer
        2 = D-pad Up modifier layer
        3 = D-pad Down modifier layer
    """
    if state['select_held']:
        return 1
    if state['dpad_up_held']:
        return 2
    if state['dpad_down_held']:
        return 3
    return 0

def get_layered_button(button_code):
    """
    Translate a physical button into its virtual alternate button,
    according to the currently active modifier layer.

    Returns None if the button should retain its normal meaning.
    """

    if button_code not in LAYER_BUTTONS:
        return None

    index = LAYER_BUTTONS.index(button_code)
    layer = get_button_layer()

    if layer == 1:
        # Select layer: EXTRA 1..9
        return EXTRA_BUTTONS[index]

    if layer == 2:
        # D-pad Up layer: EXTRA 10..18
        return EXTRA_BUTTONS[index + 9]

    if layer == 3:
        # D-pad Up layer: EXTRA 19..27
        return EXTRA_BUTTONS[index + 18]

    return None

def main():
    args = sys.argv[1:]
    if not (2 <= len(args) <= 3):
        sys.exit("usage: sudo python3 bankpad.py <padA-event> <padB-event> [virtual-name]")

    pathA, pathB = args[0], args[1]
    virtual_name = args[2] if len(args) == 3 else DEFAULT_NAME

    # trigger magnitudes
    a_l2 = a_r2 = 0.0   # pad A triggers
    b_l2 = b_r2 = 0.0   # pad B triggers
    last_pitch = last_yaw = last_bank = last_throttle = last_vertical_slide = last_horizontal_slide =  0

    last_normalized_vertical_slide = last_normalized_horizontal_slide = 0.0
    
    #for mixing of stick and gyro values
    stick_yaw = 0.0
    stick_pitch = 0.0

    stick_horizontal_slide = 0
    stick_vertical_slide = 0

    def emit_pitch():
        nonlocal last_pitch

        #stick = normalize_axis(stick_pitch, TRIGGER_MAX)
        # stick = expo_pure_jcurve(stick, 5.0)  # Apply exponential curve to gyro yaw
         
        out = int(stick_pitch) #* (-1 if PITCH_INVERT else 1))
        #out = mix_stick_gyro(stick_pitch, gyro_pitch, 255)
        # Pack the final merged pitch value onto the DS4's designated pitch axis.
        #if out != last_pitch:
        ui.write(ecodes.EV_ABS, ecodes.ABS_RY, out)
        last_pitch = out
        return True

    def emit_yaw():
        nonlocal last_yaw

        #stick = normalize_axis(stick_yaw, TRIGGER_MAX)
        #stick = expo_pure_jcurve(stick, 5.0)  # Apply exponential curve to gyro yaw
        out = int(stick_yaw)
        #print("emit_yaw:", out)
        #out = mix_stick_gyro(stick_yaw, gyro_yaw, 255)
        # Pack the final merged yaw value onto the DS4's designated yaw axis.
        #if out != last_yaw:
        ui.write(ecodes.EV_ABS, ecodes.ABS_RX, out)
        # print("emit_yaw:", out)
        last_yaw = out
        return True

    def emit_bank():
        nonlocal last_bank
        bank_out = (b_l2 + b_r2)
        #print("INICIO bank_out:", bank_out)

        #Normalize before applying deadzone
        #out = apply_deadzone(out, 0.01)  # Apply a small deadzone to avoid jitter   
        bank_out = normalize_axis(bank_out, 3200)  # Normalize to -1.0..+1.0

        bank_out = bank_out * (-1 if BANK_INVERT else 1) 
        bank_out = apply_deadzone(bank_out, 0.085) #* BANK_SENSITIVITY

        bank_out = expo_multiplier_blend (bank_out, 0.50)
        bank_out = denormalize_joystick_axis_to_8bits(bank_out)
        #print("bank_out:", bank_out)
        
        #bank_out = max(3200, min(3200, bank_out))  # Clamp to valid range 

        if bank_out != last_bank:
            ui.write(ecodes.EV_ABS, BANK_AXIS, int(bank_out))
            last_bank = bank_out
            return True
        return False

    def emit_throttle():
        nonlocal last_throttle
        # print("b_l2:", b_l2, "b_r2:", b_r2)
        out = merge(a_l2, a_r2, THROTTLE_INVERT)
        if out != last_throttle:
            ui.write(ecodes.EV_ABS, THROTTLE_AXIS, out)
            last_throttle = out 
            return True
        return False   

    def emit_vertical_slide():
        nonlocal last_vertical_slide
        out = stick_vertical_slide
        if out != last_vertical_slide:
            ui.write(ecodes.EV_ABS, ecodes.ABS_Y, out)
            last_vertical_slide = out
            return True
        return False

    def emit_horizontal_slide():
        nonlocal last_horizontal_slide
        out = stick_horizontal_slide
        if out != last_horizontal_slide:
            ui.write(ecodes.EV_ABS, ecodes.ABS_X, out)
            last_horizontal_slide = out
            return True
        return False

    def normalize_axis(value, max_value=32767):
        return max(-1.0, min(1.0, value / max_value))
    def denormalize_axis(value, max_value=32767):
        return max(-max_value, min(max_value, int(value * max_value)))

    srcA = evdev.InputDevice(pathA)
    srcB = evdev.InputDevice(pathB)
    print("Pad (primary + throttle): ", srcA.path, "->", srcA.name)
    print("Pad (Gyro controls):  ", srcB.path, "->", srcB.name)

    cap = build_caps_padA(srcA)
    ui = UInput(cap, name=virtual_name, version=0x1)
    print("Virtual pad '%s' created." % virtual_name)
    print("  throttle axis:", ecodes.ABS.get(THROTTLE_AXIS, THROTTLE_AXIS))
    print("  bank axis    :", ecodes.ABS.get(GYRO_BANK_AXIS, GYRO_BANK_AXIS))

    srcA.grab()
    srcB.grab()

    fds = {srcA.fd: srcA, srcB.fd: srcB}

    try:
        while True:
            r, _, _ = select.select(fds, [], [])
            for fd in r:
                dev = fds[fd]
                for e in dev.read():
                    if dev is srcA:
                        # Select and D-pad Up/Down are pure modifiers. They are
                        # consumed here and never passed through as ordinary input.
                        if e.type == ecodes.EV_KEY and e.code == MOD_SELECT:
                            state['select_held'] = (e.value != 0)
                            #Manejo de afterburner para cambiar dinámicamente con el botón de SELECT
                            #A diferencia de las demás acciones (Firing por ejemplo, cuando estás disparando se ingigboraora el botón select)
                            # active = state["active_layered_buttons"]
                            # if state['select_held'] and a_r2 == 0 and a_l2 > 0: #We are going forward
                            #     print("OHHHH")
                            #     button = active.get(712)                                  
                            #     if button is None:
                            #         print("AAAAAHHHH")
                            #         active[712] = 712                                    
                            #         ui.write(ecodes.EV_KEY, 712, 1)
                            # else:
                            #     button = active.pop(712, 712) 
                            #     if button is not None:
                            #         print("BUTTON:", button)
                            #         ui.write(ecodes.EV_KEY, button, 0)
                            # ui.syn()                                                
                            continue     
                        if e.type == ecodes.EV_ABS and e.code == MOD_DPAD_UP_AXIS:

                            # Typical evdev hat encoding:
                            # -1 = Up
                            #  0 = neutral
                            # +1 = Down
                            state['dpad_up_held'] = (e.value == -1)
                            state['dpad_down_held'] = (e.value == 1)
                            # Consume only the UP state if you don't want D-pad Up
                            # itself reaching DXX.
                            #
                            # For now, consume the entire Y hat axis.
                            continue
                        if e.type == ecodes.EV_ABS and e.code == MOD_DPAD_DOWN_AXIS:

                            # Typical evdev hat encoding:
                            # -1 = Up
                            #  0 = neutral
                            # +1 = Down
                            state['dpad_up_held'] = (e.value == 1)
                            state['dpad_down_held'] = (e.value == -1)
                            continue
                        if e.type == ecodes.EV_ABS and e.code == SNIPER_MODE_BUTTON:
                            if e.value == 1:
                                if state['sniper_mode_active']:
                                    if last_yaw != 0:
                                        stick_yaw = last_yaw / SNIPING_SENSITIVITY_FACTOR
                                        emit_yaw()
                                        ui.syn()  
                                    if last_pitch != 0:    
                                        stick_pitch = last_pitch / SNIPING_SENSITIVITY_FACTOR
                                        emit_pitch()
                                        ui.syn()  
                                else:
                                    if last_yaw != 0:
                                        stick_yaw = last_yaw * SNIPING_SENSITIVITY_FACTOR
                                        emit_yaw()
                                        ui.syn()  
                                    if last_pitch != 0:    
                                        stick_pitch = last_pitch * SNIPING_SENSITIVITY_FACTOR
                                        emit_pitch()
                                        ui.syn()  

                                state['sniper_mode_active'] = not state['sniper_mode_active']    
                                                        

                            continue        
                        # pad A: triggers -> throttle; everything else passes through

                        # ------------------------------------------------------------
                        # LAYERED BUTTONS
                        # ------------------------------------------------------------
                        if e.type == ecodes.EV_KEY and e.code in LAYER_BUTTONS:
                            active = state['active_layered_buttons']

                            # BUTTON DOWN
                            if e.value == 1:

                                alternate = get_layered_button(e.code)

                                if alternate is not None:
                                    active[e.code] = alternate
                                    ui.write(ecodes.EV_KEY, alternate, 1)
                                else:
                                    # Remember that this press was normal too.
                                    active[e.code] = e.code
                                    ui.write(ecodes.EV_KEY, e.code, 1)

                                ui.syn()
                                continue


                            # BUTTON UP
                            if e.value == 0:

                                # Release exactly the button that was pressed,
                                # regardless of current modifier state.
                                output_button = active.pop(e.code, e.code)

                                ui.write(ecodes.EV_KEY, output_button, 0)
                                ui.syn()
                                continue


                            # Autorepeat or unusual EV_KEY value.
                            # Forward it using the currently active mapping if present.
                            output_button = active.get(e.code, e.code)
                            ui.write(ecodes.EV_KEY, output_button, e.value)
                            ui.syn()
                            continue

                        # ------------------------------------------------------------
                        # THROTTLE TRIGGERS
                        # ------------------------------------------------------------
                                                
                        if e.type == ecodes.EV_ABS and e.code in (L2, R2):
                            if e.code == L2: 
                                a_l2 = scale_trigger(e.value)
                                a_l2 = calibrate_axis(a_l2, actual_max = 0.75) 
                            else:
                                a_r2 = scale_trigger(e.value)
                                a_r2 = calibrate_axis(a_r2, actual_max = 0.50) 
                            if emit_throttle(): 
                                ui.syn()   
                            continue

                        elif e.type == ecodes.EV_ABS and e.code == ecodes.ABS_Y:
                            # Normalize the raw stick input to -1.0..+1.0
                            stick_vertical_slide = normalize_joystick_axis_from_8bits(e.value)
                            last_normalized_vertical_slide = stick_vertical_slide = stick_vertical_slide  * (-1 if VERTICAL_SLIDE_INVERT else 1)
                            #stick_horizontal_slide_temp = normalize_joystick_axis_from_8bits(last_horizontal_slide) 

                            # Apply radial calibration to the stick input
                            _, stick_vertical_slide = radial_to_square_translation_sliding(last_normalized_horizontal_slide, stick_vertical_slide,return_axis_x=False, return_axis_y=True)
                            
                            # Denormalize back to 16-bit range for output
                            stick_vertical_slide = denormalize_joystick_axis_to_8bits(stick_vertical_slide)
                            if emit_vertical_slide(): ui.syn()
    

                        elif e.type == ecodes.EV_ABS and e.code == ecodes.ABS_X:
                            last_normalized_horizontal_slide =  stick_horizontal_slide = normalize_joystick_axis_from_8bits(e.value)
                            #stick_vertical_slide_temp = normalize_joystick_axis_from_8bits(last_vertical_slide) 

                            # Apply radial calibration to the stick input
                            stick_horizontal_slide, _ = radial_to_square_translation_sliding(stick_horizontal_slide, last_normalized_vertical_slide,return_axis_x=True, return_axis_y=False)
                            
                            # Denormalize back to 8-bit range for output
                            stick_horizontal_slide = denormalize_joystick_axis_to_8bits(stick_horizontal_slide)
                            if emit_horizontal_slide(): ui.syn()  

                            
                        # ----------------------------------------------------
                        # RIGHT STICK: ROTATION (Combat Sensitivity Adaptive)
                        # ----------------------------------------------------                                                           
                        elif e.type == ecodes.EV_ABS and e.code == ecodes.ABS_RY:
                            stick_pitch = normalize_joystick_axis_from_8bits(e.value)
                            stick_pitch = stick_pitch * (-1 if PITCH_INVERT else 1)
                            
                            stick_pitch = calibrate_asymmetric_axis_with_deadzone(stick_pitch, -0.98, 0.90, 0)
                            #print("stick_pitch:", stick_pitch)

                            if state['sniper_mode_active'] and stick_pitch != 0: 
                                stick_pitch = stick_pitch * SNIPING_SENSITIVITY_FACTOR


                            stick_yaw_temp = normalize_joystick_axis_from_8bits(last_yaw)
                            stick_pitch, _ = apply_radial_calibration_pitch_yaw(stick_pitch,
                                                                                stick_yaw_temp, 
                                                                                (0 if state['sniper_mode_active'] else 0.15), 0.07, 
                                                                                1, 
                                                                                return_pitch_axis=True, 
                                                                                return_yaw_axis=False)
                            print("stick_pitch:", stick_pitch)
                            
                            stick_pitch = denormalize_joystick_axis_to_8bits(stick_pitch)
                            if emit_pitch(): ui.syn()
                            #ui.write(e.type, e.code, mix_stick_gyro(stick_pitch, gyro_pitch, 255))
                        elif e.type == ecodes.EV_ABS and e.code == ecodes.ABS_RX:
                            stick_yaw = normalize_joystick_axis_from_8bits(e.value)   
                            #stick_yaw = calibrate_asymmetric_axis_with_deadzone(stick_yaw, -0.97, 0.89, 0)

                             
                            if state['sniper_mode_active'] and stick_yaw != 0: 
                                stick_yaw = stick_yaw * SNIPING_SENSITIVITY_FACTOR

                            #print("INICIO stick_yaw:", stick_yaw)

                            stick_pitch_temp = normalize_joystick_axis_from_8bits(last_pitch)
                            _, stick_yaw= apply_radial_calibration_pitch_yaw(stick_pitch_temp, 
                                                                            stick_yaw, 
                                                                            (0 if state['sniper_mode_active'] else 0.15), 0.1,
                                                                            1, return_pitch_axis=False,
                                                                            return_yaw_axis=True)
                            #print("stick_pitch_temp:", stick_pitch_temp)
                       
                            #print("stick_yaw:", stick_yaw)
                            stick_yaw = denormalize_joystick_axis_to_8bits(stick_yaw)



                            if emit_yaw(): ui.syn()
                        elif e.type == ecodes.EV_SYN:
                            ui.syn()
                        else:
                            ui.write(e.type, e.code, e.value)
                    else:
                        # --------------------------------------------------------# 
                        # MOTION SENSOR INTERCEPT INTERFACE (PAD B)# --------------------------------------------------------                        if e.type == ecodes.EV_ABS: 
                        if e.code == GYRO_BANK_AXIS : 
                            if e.value == 0: 
                                continue    
                            new_value = e.value     
                            if new_value < 0: 
                                #new_value = max(new_value, AXIS_OUT_MIN) 
                                b_l2 = new_value
                                b_r2 = 0
                            else:      
                                #new_value = min(new_value, AXIS_OUT_MAX)       
                                b_r2 = new_value
                                b_l2 = 0
                            #print("antes de emit_bank: ", b_r2, b_l2)    
                            if emit_bank(): ui.syn()  
    except KeyboardInterrupt:
        pass
    finally:
        for s in (srcA, srcB):
            try: s.ungrab()
            except Exception: pass
        ui.close()
        print("\nStopped; virtual pad '%s' removed." % virtual_name)


if __name__ == "__main__":
    main()
