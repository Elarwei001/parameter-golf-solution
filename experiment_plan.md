# BPB Optimization Experiment Plan

## Overview
Systematic 6-experiment plan to optimize from 1.38 BPB → target 1.13 BPB with minimal Modal costs.

## Phase 1: Low-Cost High-Impact (3 experiments, ~$2.1)

### Experiment 1.1: EMA+SWA+QK-Gain Stack
- **Base**: dim=448 Sandwich QAT (1.3805 BPB)
- **Changes**:
  - Add EMA (decay=0.997)
  - Add SWA (every 50 steps)
  - Add QK-Gain (init=4.0) 
- **Script**: `modal_phase1_ema_swa_qk.py`
- **Target**: 1.35-1.36 BPB (-1.5%)
- **Cost**: $0.70

### Experiment 1.2: Optimized QAT Timing  
- **Base**: Best from 1.1
- **Changes**:
  - QAT start: step 1500 (vs 2000)
  - QAT ramp: 1500 steps (vs 1000)
  - BPB threshold trigger: 1.45 BPB
- **Script**: `modal_phase1_qat_optimized.py`
- **Target**: 1.33-1.34 BPB (-1.5%)
- **Cost**: $0.70

### Experiment 1.3: mHC Monitoring Baseline
- **Base**: dim=448 Sandwich (no QAT)
- **Changes**: 
  - Real-time mHC parameter logging
  - 30L layer utilization profiling
- **Script**: `modal_phase1_mhc_monitor.py`
- **Target**: Data collection for Phase 2
- **Cost**: $0.70

## Phase 2: Architecture Redesign (2 experiments, ~$1.4)

### Experiment 2.1: Efficient Sandwich + Deep Layer Optimization
- **Base**: Phase 1 best config
- **Changes**:
  - Sandwich: 2×3.0x + 26×1.2x + 2×3.0x (vs 9×3.0x)
  - Layers 20+: Remove attention, MLP-only
  - Reallocate params: dim 448→480
- **Script**: `modal_phase2_efficient_sandwich.py`
- **Target**: 1.31-1.32 BPB (-1.5%)
- **Cost**: $0.70

### Experiment 2.2: Depth Recurrence Alternative
- **Base**: Phase 1 best config  
- **Changes**:
  - 6 physical layers × 5 cycles = 30 effective
  - Universal Transformer weight sharing
  - Massive param savings → dim 448→512
- **Script**: `modal_phase2_depth_recurrence.py` 
- **Target**: 1.32-1.33 BPB (exploration)
- **Cost**: $0.70

## Phase 3: Final Optimization (1 experiment, ~$0.7)

### Experiment 3.1: Best Configuration Integration
- **Base**: Phase 2 winner
- **Changes**:
  - Integrate all effective optimizations
  - TTT hyperparameter tuning (rank=16, epochs=3)
  - Final push toward 1.13 BPB target
- **Script**: `modal_phase3_final_integration.py`
- **Target**: 1.25-1.30 BPB  
- **Cost**: $0.70

## Success Metrics & Decision Points

### Phase 1 Success Criteria:
- Any experiment achieving < 1.35 BPB proceeds to Phase 2
- mHC monitoring must show clear layer utilization patterns

### Phase 2 Success Criteria:  
- Architecture changes must beat Phase 1 baseline by >0.01 BPB
- Choose winning architecture for Phase 3

### Phase 3 Success Criteria:
- Target: Break 1.30 BPB barrier
- Stretch goal: Approach 1.25 BPB

## Cost Analysis
- **Total Modal cost**: ~$4.20 (6 × $0.70)
- **ROI target**: 1.38→1.25 BPB = 9.4% improvement
- **Cost per BPB point**: ~$32 per 0.01 BPB improvement

## Risk Mitigation
- Each experiment builds on proven base configurations
- Early phases focus on additive improvements (low risk)
- Architecture changes in Phase 2 have fallback options
- Real-time monitoring prevents wasted runs

## Timeline
- **Week 1**: Phase 1 (3 experiments, 3 days)
- **Week 2**: Phase 2 (2 experiments, 2 days) 
- **Week 3**: Phase 3 (1 experiment, 1 day)
- **Total**: ~6 calendar days