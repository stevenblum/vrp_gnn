# Visualization Package

Visualization tools for CustomTSPEnv showing edge selection, deletion, and tour construction.

## Functions

### `plot_tsp_instance()`
Static plot of TSP instance with nodes, selected edges (green), and deleted edges (red/faded).

**Usage:**
```python
from visualization.plot_tour import plot_tsp_instance

fig = plot_tsp_instance(locs, adjacency, deleted_edges=[(0,1)])
fig.savefig('tsp_plot.png')
```

### `visualize_state()`
Visualize TensorDict state from environment with metrics overlay.

**Usage:**
```python
from visualization.plot_tour import visualize_state

fig = visualize_state(td, batch_idx=0, save_path='state.png')
```

### `create_episode_gif()`
Create animated GIF showing episode progression frame-by-frame.

**Usage:**
```python
from visualization.plot_tour import create_episode_gif

# action_sequence: List[(step, action_type, node_i, node_j)]
# action_type: 0=ADD, 1=DELETE, 2=DONE
create_episode_gif(locs, action_sequence, 'episode.gif', fps=2)
```

### `create_test_episode_gif()`
Quick helper to create GIF from random valid actions.

**Usage:**
```python
from visualization.plot_tour import create_test_episode_gif

path = create_test_episode_gif(env, num_loc=10, filename='test.gif')
```

## Visualization Features

### Color Coding
- **Green edges**: Currently selected (part of tour)
- **Red dashed edges**: Previously deleted (shown faded)
- **Blue nodes**: Standard nodes
- **Green nodes**: Degree = 2 (complete)
- **Red nodes**: Degree > 2 (invalid)
- **Light blue nodes**: Degree < 2 (incomplete)

### Node Labels
Each node shows:
- Node index (0, 1, 2, ...)
- Current degree in brackets [0], [1], [2]

### Metrics Display
GIFs show:
- Step number
- Action taken (ADD/DELETE/DONE with nodes)
- Total edges in graph
- Total deletions made
- Node degree list

## Examples

See `tests/test_outputs/` for generated examples:
- `test_complete_tour.gif`: Full valid TSP tour construction
- `test_deletions.gif`: Demonstration of edge deletion and re-addition
- `test_random_episode.gif`: Random valid action sequence
- `test_plot.png`: Static plot example
- `test_state.png`: State visualization with metrics

## Integration with Tests

Visualization tests are included in `tests/test_custom_tsp_env.py`:
- 6 visualization tests automatically generate example outputs
- All tests verify file creation and visual correctness
- Outputs saved to `tests/test_outputs/`
