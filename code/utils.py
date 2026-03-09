import jax.numpy as jnp
from jax import vmap
from data_structures import State

def get_all_time_points(n_cycles, tspan, bpm):
    return jnp.linspace(0, n_cycles*(60/bpm), n_cycles*tspan)

def get_final_beat_indices(n_cycles, tspan):
    """Get the start and end indices of the final beat in a time series."""
    beat = n_cycles - 1
    start_index = beat*tspan
    end_index = start_index + tspan
    return start_index, end_index

def write_many_state_trajectory_samples_to_file(solved_states, filename):
    """Writes state trajectories to a file.
    
    Parameters
    ----------
    solved_states : State
        State object whose attributes are arrays with shape (n_samples, n_times)
    filename : str
        Name of the file to save the state trajectories to
    """
    arr = solved_states.to_array()  # This is an array with whape (n_states, n_samples, n_times)
    jnp.save(filename, arr)

def read_many_state_trajectory_samples_from_file(filename):
    """Reads state trajectories from a file, as created by `write_many_state_trajectory_samples_to_file`.
    
    Parameters
    ----------
    filename : str
        Name of the file to read the state trajectories from
    
    Returns
    -------
    solved_states : State
        State object whose attributes are arrays with shape (n_samples, n_times)
    """
    reconstruct_states = vmap(vmap(State.from_array, in_axes=1, out_axes=0), in_axes=2, out_axes=-1)
    arr = jnp.load(filename)  # This is an array with shape (n_states, n_samples, n_times)
    return reconstruct_states(arr)