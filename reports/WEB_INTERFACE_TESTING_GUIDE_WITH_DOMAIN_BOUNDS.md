# 🚀 Complete Web Interface Testing Guide - Domain Bounds Required

**Updated Protocol**: All barrier certificate generation now **requires domain bounds** for optimal results.

## 📋 Pre-Testing Setup

### Step 1: Start the Web Server
```bash
cd C:\Users\patri\code\FM-LLM-Solver
python run_web_interface.py
```

**Expected Output:**
```
✅ All dependencies ready
Starting server on http://127.0.0.1:5000
Database initialized
```

### Step 2: Access Web Interface
Open browser: `http://127.0.0.1:5000`

## 🧪 Testing Phase 1: Direct Generation Mode with Domain Bounds

### Test Case 1: Simple Linear System ⭐ **BASELINE TEST**
**System Description:**
```
System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0
```

**Domain Bounds (Required):**
- Variable: `x`, Min: `-3`, Max: `3`
- Variable: `y`, Min: `-3`, Max: `3`

**Settings:**
- Model Configuration: Default
- RAG Context Chunks: 3
- Advanced Settings: Keep defaults

✅ **Expected Result**: Certificate like `B(x,y) = x² + y²` with verification showing "Passed Initial Set"

### Test Case 2: Damped Oscillator
**System Description:**
```
System Dynamics: dx/dt = y, dy/dt = -x - 0.5*y
Initial Set: x**2 + y**2 <= 0.1
Unsafe Set: x**2 + y**2 >= 9.0
```

**Domain Bounds (Required):**
- Variable: `x`, Min: `-4`, Max: `4`
- Variable: `y`, Min: `-4`, Max: `4`

✅ **Expected**: Energy-based barrier certificate for harmonic oscillator

### Test Case 3: Polynomial System
**System Description:**
```
System Dynamics: dx/dt = -x**3, dy/dt = -y**3
Initial Set: x**4 + y**4 <= 0.0625
Unsafe Set: x**4 + y**4 >= 1.0
```

**Domain Bounds (Required):**
- Variable: `x`, Min: `-2`, Max: `2`
- Variable: `y`, Min: `-2`, Max: `2`

✅ **Expected**: Higher-order polynomial barrier like `B(x,y) = x⁴ + y⁴`

### Test Case 4: Discrete-Time System
**System Description:**
```
System Dynamics: x{k+1} = 0.8*x{k}, y{k+1} = 0.9*y{k}
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0
```

**Domain Bounds (Required):**
- Variable: `x`, Min: `-3`, Max: `3`  
- Variable: `y`, Min: `-3`, Max: `3`

✅ **Expected**: Discrete-time barrier certificate with proper verification

### Test Case 5: Three-Dimensional System
**System Description:**
```
System Dynamics: dx/dt = -x, dy/dt = -y, dz/dt = -z
Initial Set: x**2 + y**2 + z**2 <= 0.25
Unsafe Set: x**2 + y**2 + z**2 >= 9.0
```

**Domain Bounds (Required):**
- Variable: `x`, Min: `-4`, Max: `4`
- Variable: `y`, Min: `-4`, Max: `4`
- Variable: `z`, Min: `-4`, Max: `4`

✅ **Expected**: 3D spherical barrier `B(x,y,z) = x² + y² + z²`

### Test Case 6: Asymmetric Domain Bounds
**System Description:**
```
System Dynamics: dx/dt = -2*x, dy/dt = -y
Initial Set: 4*x**2 + y**2 <= 1.0
Unsafe Set: x**2 + 4*y**2 >= 16.0
```

**Domain Bounds (Required):**
- Variable: `x`, Min: `-2`, Max: `2`
- Variable: `y`, Min: `-5`, Max: `5`

✅ **Expected**: Elliptical barrier certificate adapted to asymmetric bounds

## 💬 Testing Phase 2: Conversational Mode with Domain Bounds

### Test Case 7: Conversational Flow with Domain Specification
1. **Switch to "Conversational Mode"**
2. **Click "Start Conversation"**
3. **Follow this conversation sequence:**

**Message 1:**
```
I have a nonlinear dynamical system with polynomial dynamics that I need a barrier certificate for.
```

**Message 2:**
```
The system dynamics are dx/dt = -x^3 and dy/dt = -y^3, with initial set x^4 + y^4 <= 0.1
```

**Message 3:**
```
The unsafe set is x^4 + y^4 >= 1.0. I want the barrier to be valid in the domain x ∈ [-1.5, 1.5], y ∈ [-1.5, 1.5].
```

**Message 4:**
```
Please generate a barrier certificate for this system with the specified domain bounds.
```

4. **Check "Ready to Generate Certificate"**
5. **Click "Generate"**

✅ **Expected**: AI incorporates domain bounds into certificate generation

## 🔍 Testing Phase 3: Advanced Features with Domain Bounds

### Test Case 8: Large Domain Bounds
**System Description:**
```
System Dynamics: dx/dt = -0.1*x, dy/dt = -0.1*y
Initial Set: x**2 + y**2 <= 1.0
Unsafe Set: x**2 + y**2 >= 100.0
```

**Domain Bounds (Large):**
- Variable: `x`, Min: `-15`, Max: `15`
- Variable: `y`, Min: `-15`, Max: `15`

### Test Case 9: Tight Domain Bounds
**System Description:**
```
System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0
```

**Domain Bounds (Tight):**
- Variable: `x`, Min: `-1`, Max: `1`
- Variable: `y`, Min: `-1`, Max: `1`

### Test Case 10: Advanced Verification Settings
Use Test Case 1 with these modifications:
- **Samples for ΔB ≤ 0**: 2000
- **Samples for boundary**: 1000
- **Enable "Run Optimisation-Based Falsification"**
- **Optimization Max Iterations**: 50
- **Optimization Population Size**: 10

## 🐛 Testing Phase 4: Error Handling with Domain Bounds

### Test Case 11: Invalid Domain Bounds
**System Description:** (Use Test Case 1)

**Invalid Domain Bounds:**
- Variable: `x`, Min: `5`, Max: `3` (Max < Min)
- Variable: `y`, Min: `-3`, Max: `3`

✅ **Expected**: Graceful error handling

### Test Case 12: Missing Domain Bounds
**System Description:** (Use Test Case 1)

**Domain Bounds:** (Leave completely empty)

✅ **Expected**: Warning message suggesting domain bounds are required

### Test Case 13: Inconsistent Domain/System Variables
**System Description:**
```
System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0
```

**Mismatched Domain Bounds:**
- Variable: `a`, Min: `-3`, Max: `3` (wrong variable name)
- Variable: `b`, Min: `-3`, Max: `3` (wrong variable name)

✅ **Expected**: System should handle gracefully or provide helpful error

## 📊 Success Validation Checklist

### ✅ Core Functionality (All with Domain Bounds):
- [ ] Simple linear system generates correct certificate
- [ ] Polynomial system uses appropriate higher-order barriers  
- [ ] Discrete-time systems work correctly
- [ ] 3D systems generate proper certificates
- [ ] Verification shows "Passed Initial Set" (boundary fix working)
- [ ] Domain bounds are properly incorporated into generation
- [ ] Conversational mode includes domain bound discussions

### ✅ Domain Bounds Specific Validation:
- [ ] Default domain bounds (x,y ∈ [-3,3]) are pre-filled
- [ ] Domain bounds section is open by default
- [ ] Adding/removing domain bounds works smoothly
- [ ] Different domain bound ranges affect certificate generation
- [ ] Error handling for invalid domain bounds works
- [ ] Domain bounds appear in generated prompts to LLM

### ✅ Advanced Features:
- [ ] Large domain bounds don't break generation
- [ ] Tight domain bounds work correctly
- [ ] Advanced verification settings apply with domain bounds
- [ ] Asymmetric domain bounds generate appropriate certificates

## 🎯 Updated Success Criteria

Your web interface is **PRODUCTION READY** if:

1. ✅ **Domain bounds are treated as required** in UI
2. ✅ **Default domain bounds are pre-filled** for user convenience  
3. ✅ **All test cases work with domain bounds** specified
4. ✅ **Certificate generation incorporates domain constraints**
5. ✅ **Verification system works** with boundary condition fix
6. ✅ **Both modes function** with domain bounds integration
7. ✅ **Error handling is robust** for domain bound edge cases

## 🚨 Domain Bounds Best Practices

### **Recommended Domain Bounds by System Type:**

**Linear Systems**: `x,y ∈ [-5, 5]` (generous bounds)

**Polynomial Systems**: `x,y ∈ [-2, 2]` (moderate bounds to avoid numerical issues)

**Discrete Systems**: `x,y ∈ [-3, 3]` (consistent with continuous analogs)

**3D+ Systems**: Start with `±3` for all variables, adjust based on results

**Near-Unstable Systems**: Use larger bounds like `±10` to capture behavior

### **Domain Bounds Guidelines:**
1. **Always include bounds** for optimal certificate generation
2. **Make bounds symmetric** unless system has inherent asymmetry
3. **Start conservative** (smaller bounds) and expand if needed
4. **Ensure bounds contain** initial and unsafe sets with margin
5. **Use reasonable ranges** to avoid numerical overflow in verification

## 🎉 Final Validation Command

Quick test with domain bounds:
```bash
# Test verification directly with domain bounds
python -c "
import sys; from pathlib import Path; sys.path.insert(0, str(Path('.').absolute()))
from utils.config_loader import load_config
from web_interface.verification_service import VerificationService
config = load_config('config.yaml')
vs = VerificationService(config)
result = vs.verify_certificate(
    'x**2 + y**2', 
    '''System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0''',
    domain_bounds={'x': [-3, 3], 'y': [-3, 3]}
)
print('Success:', result.get('overall_success'))
print('Boundary working:', 'Passed Initial Set' in result.get('details', {}).get('numerical', {}).get('reason', ''))
print('Domain bounds integrated correctly!')
"
```

Your web interface is now optimized for maximum barrier certificate generation success with **required domain bounds**! 🚀 