# Troubleshooting Guide

## Data Collection Issues

### Connection Reset Errors

**Problem**: Getting `ConnectionResetError` or `Connection aborted` errors when collecting data from Riot API.

**Cause**: Network instability, Riot API rate limiting, or temporary connection issues.

**Solution**: The script now includes automatic retry logic with exponential backoff. If you still see errors:

1. **Reduce the collection target** (for testing):
   ```python
   # In collect_from_leaderboard.py, line ~339-345
   # Change from:
   all_data.extend(collect_from_region("kr", target_jungle_matches=800, matches_per_player=40))

   # To (smaller target):
   all_data.extend(collect_from_region("kr", target_jungle_matches=100, matches_per_player=20))
   ```

2. **Increase the delay between requests**:
   ```python
   # In collect_from_leaderboard.py, line ~31
   # Change from:
   REQUEST_DELAY = 0.05  # 20 req/sec

   # To (slower):
   REQUEST_DELAY = 0.1  # 10 req/sec
   ```

3. **Check your API key validity**:
   - Go to https://developer.riotgames.com/
   - Verify your API key hasn't expired
   - Development keys expire after 24 hours
   - Request a production key for longer collection sessions

4. **Run collection in smaller batches**:
   ```bash
   # Instead of collecting all at once, collect one region at a time:

   # First, edit collect_from_leaderboard.py to only collect from NA:
   # Comment out KR and EUW lines in main()
   python scripts/collect_from_leaderboard.py

   # Then collect from EUW, etc.
   ```

### API Rate Limiting

**Problem**: Getting 429 (Too Many Requests) errors.

**Solution**:
- The script already handles 429 errors with automatic retries
- If persistent, increase `REQUEST_DELAY` to 0.1 or 0.2 seconds
- Request a production API key from Riot for higher rate limits

### Timeout Errors

**Problem**: Requests timing out.

**Solution**: The script uses 10-second timeouts. If your connection is slow:

```python
# In each API function, change timeout:
response = _session.get(url, headers=headers, timeout=30)  # Increase to 30s
```

## Model Training Issues

### Out of Memory

**Problem**: Training crashes with out-of-memory errors.

**Solution**:
```python
# In train_behavior_cloning.py, reduce batch size:
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Was 16
```

### Low Accuracy

**Problem**: BC model accuracy < 50%.

**Possible causes**:
1. Not enough training data - collect more matches
2. Data quality issues - check `data/processed/` files
3. State representation mismatch - ensure powerspike data is aligned

**Debug**:
```python
# Check data shapes:
python src/training_data.py

# Should show:
# State dimension: 83 (with powerspikes) or 73 (without)
# Check that states.npy and actions.npy have matching first dimension
```

### Gank Priority Model Low Accuracy

**Problem**: Random Forest accuracy < 55%.

**Possible causes**:
1. Lane assignment heuristic failing (participant ID ≠ actual role)
2. Not enough gank examples in dataset
3. Powerspike data quality

**Solution**:
- Collect more data with more gank actions
- Check that powerspike data has lane snapshots
- Inspect training examples manually in `train_gank_priority_model.py`

## Runtime Errors

### ModuleNotFoundError: gank_priority

**Problem**: `ImportError: cannot import name 'GankPriorityPredictor'`

**Solution**:
- This is expected if you haven't trained the gank priority model yet
- The code will fall back to heuristic-based priorities
- Train the model: `python scripts/train_gank_priority_model.py`

### File Not Found Errors

**Problem**: Can't find `challenger_jungle_data.json` or other data files.

**Solution**: Run the data collection pipeline in order:
```bash
# Step 1: Collect jungle matches
python scripts/collect_from_leaderboard.py

# Step 2: Collect powerspike data
python scripts/collect_powerspike_data.py

# Step 3: Train models
python scripts/train_gank_priority_model.py
```

### Dimension Mismatch

**Problem**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Cause**: State dimension doesn't match model input dimension.

**Solution**:
- Check your model's expected input dimension
- Verify state vector creation matches expected size
- Retrain model if you changed state representation

## Performance Issues

### Data Collection Too Slow

**Problem**: Taking >6 hours to collect data.

**Solution**:
1. **Reduce targets**: Collect 1000 matches instead of 2000
2. **Parallelize**: Run collection for different regions in parallel
3. **Use production API key**: Higher rate limits

### Training Too Slow

**Problem**: BC/RL training taking too long.

**Solution**:
```python
# Reduce epochs:
n_epochs = 20  # Instead of 50

# Use GPU if available:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reduce model size:
model = JungleNet(state_dim=83, action_dim=20, hidden_dim=64)  # Was 128
```

## Common Mistakes

### 1. Forgot to Run Powerspike Collection

**Symptom**: BC model trains but doesn't use powerspike features.

**Fix**: Make sure you run `python scripts/collect_powerspike_data.py` after collecting jungle data.

### 2. Using Old Data Files

**Symptom**: Model doesn't improve despite code changes.

**Fix**: Delete old `.npy` files and regenerate:
```bash
rm data/processed/states.npy
rm data/processed/actions.npy
rm data/processed/game_indices.npy
python src/training_data.py
```

### 3. API Key Expired

**Symptom**: All API requests fail with 403 Forbidden.

**Fix**:
- Development keys expire after 24 hours
- Get a new key from https://developer.riotgames.com/
- Update `API_KEY` in `collect_from_leaderboard.py` and `collect_powerspike_data.py`

## Getting Help

If you're still stuck:

1. **Check the error message carefully** - often it tells you exactly what's wrong
2. **Look at the data files** - open `.json` files to verify structure
3. **Test components individually**:
   ```bash
   python src/gank_priority.py  # Test gank predictor
   python src/powerspike_system.py  # Test powerspike system
   python src/jungle_rl_env.py  # Test RL environment
   ```

4. **Start small**: Test with 10-20 matches first, then scale up

## Quick Fixes for Common Errors

```bash
# Connection reset during data collection:
# → Just re-run the script, it will skip already collected data

# Out of memory during training:
# → Reduce batch size in training script

# Can't import gank_priority:
# → It's okay, fallback will be used. Train the model when ready.

# Wrong state dimension:
# → Retrain the BC model after fixing state representation

# Model not learning:
# → Check that you have enough data (>500 examples minimum)
```

## Prevention

To avoid issues:

✅ **Start with small data collection** (100 matches) to test the pipeline
✅ **Verify each step** before moving to the next
✅ **Keep backups** of successfully collected data
✅ **Use version control** to track code changes
✅ **Test on CPU** before scaling to GPU
✅ **Monitor disk space** - data files can be large
